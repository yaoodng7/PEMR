import torch
import torch.nn as nn
from .model import Model
import torchaudio


class Conv_1d(Model):
    def __init__(self, input_channels, output_channels, shape=3, stride=3, pooling=2):
        super(Conv_1d, self).__init__()
        self.conv = nn.Conv1d(
            input_channels, output_channels, shape, stride=stride, padding=1
        )
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Fusion_encoder(Model):

    def __init__(self, args, out_dim):
        super(Fusion_encoder, self).__init__()
        self.args = args
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm1d(args.n_mels)

        # FCN
        self.layer1 = Conv_1d(args.n_mels, 256)
        self.layer2 = Conv_1d(256, 256)
        self.layer3 = Conv_1d(256, 256)
        self.layer4 = Conv_1d(256, 512)
        #self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        # # Dense
        # self.dense = nn.Linear(64, n_class)
        # self.dropout = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.args.encoder_dim, out_dim)

    def forward(self, x):
        # Spectrogram
        x = self.to_db(x)
        x = self.spec_bn(x)
        # FCN
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        #x = torch.cat([x1.mean(dim=-1), x2.mean(dim=-1), x3.mean(dim=-1)], dim=-1)
        #x = self.layer5(x)
        # Dense
        #x = self.fc(x)
        #x = self.dropout(x)
        # x = self.dense(x)
        # x = self.sigmoid(x)

        #[b_size, 57, 512]
        return x4.squeeze()
