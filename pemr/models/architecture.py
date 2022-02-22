import torch.nn as nn
import torchaudio
from .model import Model
import torch
from .gen_and_pred import UnitedModule

class Conv_2d(Model):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out



class Architecture(Model):
    def __init__(self, args):
        super(Architecture, self).__init__()
        self.args = args
        self.spec_bn = nn.BatchNorm2d(1)

        # mask generation module and predicting module
        self.unitedmodule = UnitedModule(args)

        # encoder--FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2, 4))
        self.layer2 = Conv_2d(64, 128, pooling=(2, 4))
        self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 5))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x

    def trm_mask(self, x_i, x_j):

        # generate augmented positives and negatives and prediction loss
        x_pos, x_neg, loss_pred = self.unitedmodule.gen_mask(x_i.transpose(1, 2), x_j.transpose(1, 2))

        h_i, h_pos, h_neg = self.forward(x_i), self.forward(x_pos), self.forward(x_neg)
        return  h_i, h_pos, h_neg, loss_pred

