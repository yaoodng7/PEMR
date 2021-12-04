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
from pemr.models import SampleCNN
from pemr.modules import ContrastiveLearning, LinearEvaluation
from pemr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)
from pemr.models import FCN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    args.accelerator = None

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # ------------
    # dataloaders
    # ------------
    train_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="train")
    valid_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="valid")
    test_dataset = get_dataset(args.dataset, args.dataset_dir, args.label_factor, subset="test")

    contrastive_train_dataset = ContrastiveDataset(
        train_dataset,
        input_shape=(1, args.audio_length),
        transform=Compose(train_transform),
    )

    contrastive_valid_dataset = ContrastiveDataset(
        valid_dataset,
        input_shape=(1, args.audio_length),
        transform=Compose(train_transform),
    )

    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
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
    encoder = FCN(args, out_dim=train_dataset.n_classes)
    args.accelerator = 'gpu'
    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

    state_dict = load_encoder_checkpoint(args.checkpoint_path)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    module = LinearEvaluation(
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
        args.audio_length,
        device=device,
    )
    print(results)
