import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pemr.models import Architecture
# Audio Augmentations
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    Reverb,
)
import os
from pemr.data import ContrastiveDataset
from pemr.datasets import get_dataset
from pemr.modules import ContrastiveLearning

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # data augmentations
    train_transform = [
        RandomResizedCrop(n_samples=args.segment_length),
        RandomApply([PolarityInversion()], p=args.polarity),
        RandomApply([Noise()], p=args.noise),
        RandomApply([Gain()], p=args.gain),
        RandomApply(
            [HighLowPass(sample_rate=args.sample_rate)], p=args.filters
        ),
        RandomApply([Delay(sample_rate=args.sample_rate)], p=args.delay),
        RandomApply(
            [Reverb(sample_rate=args.sample_rate)], p=args.reverb
        ),
    ]
    num_augmented_samples = 2

    # dataloaders
    train_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="valid")
    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.segment_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.segment_length),
        transform=ComposeMany(
            train_transform, num_augmented_samples=num_augmented_samples
        ),
    )

    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )

    valid_loader = DataLoader(
        contrastive_valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )

    # architecture
    encoder = Architecture(args)
    args.accelerator = 'gpu'

    # model
    module = ContrastiveLearning(args, encoder)
    logger = TensorBoardLogger("runs", name="PEMR-{}".format(args.dataset))
    if args.checkpoint_path:
        module = module.load_from_checkpoint(
            args.checkpoint_path, encoder=encoder, output_dim=train_dataset.n_classes
        )

    else:
        # training
        trainer = Trainer.from_argparse_args(
            args,
            logger=logger,
            sync_batchnorm=True,
            max_epochs=args.max_epochs,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            accelerator=args.accelerator,
        )
        trainer.fit(module, train_loader, valid_loader)