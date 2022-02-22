import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from pemr.datasets import get_dataset
from pemr.data import ContrastiveDataset
from pemr.evaluation import evaluate
from pemr.modules import ContrastiveLearning, LinearEvaluation, Finetuner
from pemr.utils import (
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)
from pemr.models import Architecture

# configuration
# -----------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='1')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--dataset_dir', type=str, default='../WWWv2/data/')
parser.add_argument('--seed', type=int, default=42)
# pre-training config
parser.add_argument('--projection_dim', type=int, default=256)
parser.add_argument('--encoder_dim', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--dataset', type=str, default='magnatagatune', help='magnatagatune or gtzan' )
parser.add_argument('--clips_num', type=int, default=462)
parser.add_argument('--n_heads', type=int, default=3)
parser.add_argument('--transformer_encoder_layers', type=int, default=3)
parser.add_argument('--label_factor', type=float, default=1.0)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=3.0e-4)
parser.add_argument('--weight_decay', type=float, default=1.0e-6)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--lambd1', type=float, default=5.0e-3)
parser.add_argument('--lambd2', type=float, default=5.0e-3)

# data augmentations config
parser.add_argument('--segment_length', type=int, default=59049)
parser.add_argument('--sample_rate', type=int, default=22050)
parser.add_argument('--polarity', type=float, default=0.8)
parser.add_argument('--noise', type=float, default=0.01)
parser.add_argument('--gain', type=float, default=0.3)
parser.add_argument('--filters', type=float, default=0.8)
parser.add_argument('--delay', type=float, default=0.3)
parser.add_argument('--reverb', type=float, default=0.6)

# fine-tuning config
parser.add_argument('--finetune', type=int, default=0)
parser.add_argument('--finetuner_mlp', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--finetuner_checkpoint_path', type=str, default='')
parser.add_argument('--finetuner_max_epochs', type=int, default=300)
parser.add_argument('--finetuner_batch_size', type=int, default=64)
parser.add_argument('--finetuner_learning_rate', type=int, default=0.003)

# masking config
parser.add_argument('--masked_factor', type=float, default=0.1)

# mel-spectrogram config
parser.add_argument('--hop_size', type=int, default=128)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--n_fft', type=int, default=256)
# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    args.accelerator = None

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.segment_length)]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="test")

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.segment_length),
        transform=Compose(train_transform),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.segment_length),
        transform=Compose(train_transform),
    )

    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.segment_length),
        transform=None,
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=True,
    )

    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    test_loader = DataLoader(
        contrastive_test_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    # ------------
    # encoder
    # ------------
    encoder = Architecture(args)
    args.accelerator = 'gpu'
    n_features = args.encoder_dim

    state_dict = load_encoder_checkpoint(args.checkpoint_path)
    encoder.load_state_dict(state_dict)
    cl = ContrastiveLearning(args, encoder)

    # line evaluation or fine-tuning
    if args.finetune == 0:
        cl.eval()
        cl.freeze()
        module = LinearEvaluation(
            args,
            cl.encoder,
            hidden_dim=n_features,
            output_dim=train_dataset.n_classes,
        )
    elif args.finetune == 1:
        module = Finetuner(
            args,
            cl.encoder,
            hidden_dim=n_features,
            output_dim=train_dataset.n_classes,
        )

    if args.finetuner_checkpoint_path:
        state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
        module.model.load_state_dict(state_dict)
    else:
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=10, verbose=False, mode="min"
        )

        trainer = Trainer.from_argparse_args(
            args,
            logger=TensorBoardLogger(
                "runs", name="PEMR-eval-{}".format(args.dataset)
            ),
            max_epochs=args.finetuner_max_epochs,
            callbacks=[early_stop_callback],
            accelerator = args.accelerator
        )
        trainer.fit(module, train_loader, valid_loader)

    device = "cuda:0" if args.gpus else "cpu"
    results = evaluate(
        args,
        module.encoder,
        module.model,
        contrastive_test_dataset,
        args.dataset,
        args.segment_length,
        device=device,
    )
    print(results)
