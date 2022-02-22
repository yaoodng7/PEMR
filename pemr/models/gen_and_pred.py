from pemr.models import Model
import torch
from .embedding import PositionalEmbedding
from .transformer_encoder import PoswiseFeedForwardNet
from .mask import Mask


class UnitedModule(Model):
    def __init__(self, args):
        super(UnitedModule, self).__init__()
        self.args = args

        self.clips_num = self.args.clips_num
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.transformer_encoder_layers
        self.batch_size = self.args.batch_size
        self.n_features = 128
        self.masked_num = int(self.clips_num * self.args.masked_factor)

        # predicting layer
        self.pred_layer = PoswiseFeedForwardNet(128, 512)

        self.index = torch.tensor([0])
        self.bias = []
        for i in range(self.batch_size):
            self.bias += [self.clips_num * i] * self.masked_num
        self.bias = torch.tensor(self.bias)

        self.CLS_emb = torch.nn.Embedding(1, self.n_features)
        self.POS_emb = PositionalEmbedding(self.n_features, self.clips_num + 1).pe
        self.MASK_emb = torch.nn.Embedding(1, self.n_features)
        self.masking = Mask(self.n_layers, 128, 128, self.n_features,
                            self.n_heads, self.clips_num, self.bias, self.masked_num)

    def huber(self, true, pred, delta=0.1):
        loss = torch.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                           delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
        return torch.mean(loss)

    def predicting(self, x_i, x_j):
        # ------------
        # same as bert
        # ------------
        random_mask_i = torch.bernoulli(torch.full((self.batch_size, self.clips_num + 1), 0.15)).unsqueeze(2)
        random_mask_j = torch.bernoulli(torch.full((self.batch_size, self.clips_num + 1), 0.15)).unsqueeze(2)
        random_mask_i = random_mask_i.repeat(1, 1, self.n_features).cuda()
        random_mask_j = random_mask_j.repeat(1, 1, self.n_features).cuda()

        mask_strategy = torch.tensor([0.8, 0.1, 0.1])
        strategy_id = torch.multinomial(mask_strategy, 1)
        if strategy_id == 0:
            pred_i = x_i * (1 - random_mask_i)
            pred_j = x_j * (1 - random_mask_j)
        elif strategy_id == 1:
            idx = torch.randint(self.clips_num + 1, (1,))
            replacing_i = x_i[:, idx:idx + 1, :].repeat(1, self.clips_num + 1, 1)
            replacing_j = x_j[:, idx:idx + 1, :].repeat(1, self.clips_num + 1, 1)
            pred_i = x_i * (1 - random_mask_i) + random_mask_i * replacing_i
            pred_j = x_j * (1 - random_mask_j) + random_mask_j * replacing_j
        elif strategy_id == 2:
            pred_i = x_i
            pred_j = x_j

        pred_i = self.pred_layer(self.masking.trm(pred_i))
        pred_j = self.pred_layer(self.masking.trm(pred_j))
        re_loss = self.huber(x_i * random_mask_i, pred_i * random_mask_i) + \
                  self.huber(x_j * random_mask_j, pred_j * random_mask_j)

        return re_loss

    def gen_mask(self, x_i, x_j):
        self.index = self.index.cuda()
        CLS = self.CLS_emb(self.index)
        MASK = self.MASK_emb(self.index)
        batch_CLS = CLS.repeat(self.batch_size, 1, 1).cuda()
        batch_POS = self.POS_emb.repeat(self.batch_size, 1, 1).cuda()
        batch_MASK = MASK.repeat(self.batch_size, self.clips_num, 1).cuda()
        m_i = torch.cat((batch_CLS, x_i), dim=1)
        m_j = torch.cat((batch_CLS, x_j), dim=1)
        m_i = m_i + batch_POS
        m_j = m_j + batch_POS

        # predicting
        loss_pred = self.predicting(m_i, m_j)
        mask_j = self.masking(m_i, m_j)
        x_pos = x_j * mask_j + (1 - mask_j) * batch_MASK
        x_neg = x_j * (1 - mask_j) + mask_j * batch_MASK

        return x_pos.transpose(1, 2), x_neg.transpose(1, 2), loss_pred


