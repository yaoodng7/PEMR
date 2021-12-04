import torch.nn as nn
import torchaudio
from .model import Model
import torch
from .embedding import PositionalEmbedding
from .mask import Mask

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



class FCN(Model):
    """
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    """

    def __init__(self, args, out_dim):
        super(FCN, self).__init__()
        self.args = args
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.clips_num = self.args.clips_num
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.transformer_encoder_layers
        self.batch_size = self.args.batch_size
        self.n_features = 128
        self.masked_num = int(self.clips_num * self.args.masked_factor)

        self.index = torch.tensor([0]).cuda()
        self.bias = []
        for i in range(self.batch_size):
            self.bias += [self.clips_num * i] * self.masked_num
        self.bias = torch.tensor(self.bias).cuda()

        self.CLS_emb = torch.nn.Embedding(1, self.n_features).cuda()
        self.POS_emb = PositionalEmbedding(self.n_features, self.clips_num + 1).pe.cuda()
        self.MASK_emb = torch.nn.Embedding(1, self.n_features).cuda()

        self.masking = Mask(self.n_layers, 128, 128, self.n_features,
                            self.n_heads, self.clips_num, self.bias, self.masked_num)
        # self.reconstruct = Decoder(self.hparams.encoder_channels)
        # self.criterion_B = self.configure_criterion()
        # self.mse_loss = nn.MSELoss()+
        self.step = 0
        self.mask_weight = args.mask_weight
        #self.mask_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.result = []


        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2, 4))
        self.layer2 = Conv_2d(64, 128, pooling=(2, 4))
        self.layer3 = Conv_2d(128, 128, pooling=(2, 4))
        self.layer4 = Conv_2d(128, 128, pooling=(4, 5))
        self.fc = nn.Linear(self.args.encoder_dim, out_dim)

    def forward(self, x):
        # Spectrogram
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        #x = self.layer5(x)
        # Dense
        #x = self.fc(x)
        #x = self.dropout(x)
        # x = self.dense(x)
        # x = self.sigmoid(x)

        return x

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        return x



    def trm_mask(self, x_i, x_j):
        b_size = x_i.size(0)
        x_i = self.to_db(x_i)
        x_j = self.to_db(x_j)


        x_i = x_i.squeeze().transpose(1, 2)
        x_j = x_j.squeeze().transpose(1, 2)

        CLS = self.CLS_emb(self.index)
        MASK = self.MASK_emb(self.index)
        #
        batch_CLS = CLS.repeat(b_size, 1, 1)
        batch_POS = self.POS_emb.repeat(b_size, 1, 1)
        batch_MASK = MASK.repeat(b_size, self.clips_num, 1)
        #
        m_i = torch.cat((batch_CLS, x_i), dim=1)
        m_j = torch.cat((batch_CLS, x_j), dim=1)
        m_i = m_i + batch_POS
        m_j = m_j + batch_POS

        mask_j, re_loss = self.masking(m_i, m_j)
        # if self.step < 15000:
        #     self.step += 1
        #     return re_loss
        #
        if self.step >= -1:
            x_j = x_j * (mask_j + (1 - mask_j)*self.mask_weight) + (1 - mask_j)*(1 - self.mask_weight)*batch_MASK
            x_neg = x_j * (1 - mask_j) + mask_j * batch_MASK
        # x_j = x_j * mask + (1 - mask) * bacth_MASK
        #     mask_strategy = torch.tensor([0.8, 0.1, 0.1])
        #     strategy_id = torch.multinomial(mask_strategy, 1)
        #     if strategy_id == 0:
        #         x_j = x_j * mask + (1 - mask) * bacth_MASK
        #     elif strategy_id == 1:
        #         pass
        #         # idx = torch.randint(self.clips_num, (1,))
        #         # replacing_j = x_j[:, idx:idx+1, :].repeat(1, self.clips_num, 1)
        #         # x_j = x_j * mask + replacing_j * (1 - mask)
        #     elif strategy_id == 2:
        #         pass
        h_i, h_j, h_neg = self.encode(x_i.transpose(1, 2)), self.encode(x_j.transpose(1, 2)), self.encode(x_neg.transpose(1, 2))
        self.step += 1
        return  h_i, h_j, h_neg, re_loss

