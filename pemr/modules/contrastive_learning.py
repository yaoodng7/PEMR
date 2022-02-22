import torch
from pytorch_lightning import LightningModule
from simclr.modules import LARS
import torchaudio
from .BTLoss import BTLoss
from pemr.models import Framework

class ContrastiveLearning(LightningModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.n_features = self.hparams.encoder_dim
        self.clips_num = self.hparams.clips_num
        self.n_heads = self.hparams.n_heads
        self.n_layers = self.hparams.transformer_encoder_layers
        self.batch_size = self.hparams.batch_size
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        # consists of encoder and projector
        self.model = Framework(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion_B = BTLoss(self.hparams.batch_size, self.hparams.lambd1,
                                  self.hparams.lambd2, self.hparams.projection_dim)

    def forward(self, x_i, x_j):
        x_i = self.to_db(x_i)
        x_j = self.to_db(x_j)
        z_i, z_pos, z_neg, loss_pred = self.model(x_i, x_j)
        # include loss_pos, loss_neg and loss_pred
        loss = self.criterion_B(z_i, z_pos, z_neg) + 0.01*loss_pred
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.cpu()
        x_i = x[:, 0, :].squeeze()
        x_j = x[:, 1, :].squeeze()
        x_i = torchaudio.transforms.MelSpectrogram(
            hop_length=self.hparams.hop_size, n_mels=self.hparams.n_mels, n_fft=self.hparams.n_fft)(x_i)  # (b, 96, 923)
        x_j = torchaudio.transforms.MelSpectrogram(
            hop_length=self.hparams.hop_size, n_mels=self.hparams.n_mels, n_fft=self.hparams.n_fft)(x_j)
        x_i = x_i.cuda()
        x_j = x_j.cuda()
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * self.hparams.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.hparams.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.max_epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
