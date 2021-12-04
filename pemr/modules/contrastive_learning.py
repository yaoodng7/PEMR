import torch
from pytorch_lightning import LightningModule

from simclr import SimCLR
from simclr.modules import NT_Xent, LARS
import torchaudio
from .BTLoss import BTLoss
from pemr.models import Framework
class ContrastiveLearning(LightningModule):
    def __init__(self, args, encoder):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.n_features = (
            self.encoder.fc.in_features
        )  # get dimensions of last fully-connected layer
        self.clips_num = self.hparams.clips_num
        self.n_heads = self.hparams.n_heads
        self.n_layers = self.hparams.transformer_encoder_layers
        self.batch_size = self.hparams.batch_size
        #
        # self.index = torch.tensor([0]).cuda()
        #
        # self.CLS_emb = torch.nn.Embedding(1, self.n_features).cuda()
        # self.POS_emb = PositionalEmbedding(self.n_features, self.clips_num+1).pe.cuda()
        self.model = Framework(self.encoder, self.hparams.projection_dim, self.n_features )
        # self.masking = Mask(self.n_layers, self.n_features, self.n_heads, self.clips_num)
        # self.reconstruct = Decoder(self.hparams.encoder_channels)
        self.criterion_B = self.configure_criterion()

    def forward(self, x_i, x_j):

        z_i, z_j, z_neg, re_loss = self.model(x_i, x_j)
        loss = self.criterion_B(z_i, z_j, z_neg)  + 0.01*re_loss
        # z_i_op, z_j_op = self.projector(h_i_op), self.projector(h_j_op)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.cpu()
        x_i = x[:, 0, :]
        x_j = x[:, 1, :]
        x_i = torchaudio.transforms.MelSpectrogram(
            hop_length=self.hparams.hop_size, n_mels=self.hparams.n_mels, n_fft=self.hparams.n_fft)(x_i)  # (b, 96, 923)
        x_j = torchaudio.transforms.MelSpectrogram(
            hop_length=self.hparams.hop_size, n_mels=self.hparams.n_mels, n_fft=self.hparams.n_fft)(x_j)
        x_i = x_i.squeeze().cuda()
        x_j = x_j.squeeze().cuda()
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_criterion(self):
        # PT lightning aggregates differently in DP mode
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size

        # criterion = NT_Xent(batch_size, self.hparams.temperature, world_size=1)
        criterion = BTLoss(self.hparams.batch_size, self.hparams.lambd1, self.hparams.lambd2, self.hparams.projection_dim)
        return criterion

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
