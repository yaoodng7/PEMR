import torch
import torch.nn as nn
import torchaudio
from pemr.models import Model



class DeConv_2d(Model):
    def __init__(self, input_channels, output_channels, scale_factors, shape=(3, 3), stride=1):
        super(DeConv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size=shape, stride=stride, padding=1
        )
        self.up = nn.Upsample(scale_factor=scale_factors, mode='bilinear', )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(self.up(x))))
        return out


class Decoder(Model):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.spec_bn = nn.BatchNorm2d(channels)

        # FCN
        self.layer1 = DeConv_2d(channels, 64, scale_factors=(2, 5))
        self.layer2 = DeConv_2d(64, 64, scale_factors=(2, 3))
        self.layer3 = DeConv_2d(64, 64, shape=(3, 2), scale_factors=(2, 2))
        self.layer4 = DeConv_2d(64, 1, shape=(3, 2), scale_factors=(2, 2))
        # self.layer5 = Conv_2d(128, 64, pooling=(4, 4))

        # # Dense
        # self.dense = nn.Linear(64, n_class)
        # self.dropout = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spectrogram
        x = self.spec_bn(x)
        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze()
        # x = self.layer5(x)
        # Dense
        # x = self.fc(x)
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = self.sigmoid(x)

        return x