import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import metrics
import torchaudio

class Finetuner(LightningModule):
    def __init__(self, args, encoder, hidden_dim, output_dim):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if self.hparams.finetuner_mlp:
            self.model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.model = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim))
        self.criterion = nn.BCEWithLogitsLoss()
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.accuracy = metrics.Accuracy()
        self.average_precision = metrics.AveragePrecision(pos_label=1)

    def forward(self, x, y):
        x = self.to_db(x)
        h0 = self.encoder(x)
        preds = self.model(h0)
        loss = self.criterion(preds, y)
        return loss, preds


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.cpu().squeeze()
        x = torchaudio.transforms.MelSpectrogram(
            hop_length=self.hparams.hop_size, n_mels=self.hparams.n_mels, n_fft=self.hparams.n_fft)(x)
        x = x.cuda()
        loss, preds = self.forward(x, y)
        self.log("Train/pr_auc", self.average_precision(preds, y))
        self.log("Train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.cpu().squeeze()
        x = torchaudio.transforms.MelSpectrogram(
            hop_length=self.hparams.hop_size, n_mels=self.hparams.n_mels, n_fft=self.hparams.n_fft)(x)
        x = x.cuda()
        loss, preds = self.forward(x, y)
        self.log("Valid/pr_auc", self.average_precision(preds, y))
        self.log("Valid/loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.encoder.parameters()},
             {'params': self.model.parameters()}],
            lr=self.hparams.finetuner_learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )
        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "Valid/loss",
            }
        else:
            return {"optimizer": optimizer}
