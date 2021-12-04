import torch
import torch.nn as nn
import numpy as np
from .transformer_encoder import Encoder, PoswiseFeedForwardNet
class Mask(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, n_heads, clips_num, bias, mask_num):
        super(Mask, self).__init__()
        self.encoder = Encoder(n_layers, d_k, d_v, d_model, n_heads)
        self.n_layers = n_layers
        self.clips_num = clips_num
        self.d_model = d_model
        self.scale_factor = np.sqrt(d_model)
        self.threshold = 0
        self.pred_layer = PoswiseFeedForwardNet(128, 512)
        # self.mask_embedding = nn.Embedding(2, d_model).cuda()
        # self.mask_idx = torch.tensor([0, 1]).cuda()
        self.bias = bias
        self.mask_num = mask_num

    def huber(self, true, pred, delta=0.1):

        loss = torch.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                           delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
        return torch.mean(loss)

    def trm(self, x_i):
        t_i = self.encoder(x_i)
        return t_i

    def forward(self, clips_i, clips_j):
        b_size = clips_i.size(0)
        output_i = self.encoder.dropout_emb(clips_i)
        output_j = self.encoder.dropout_emb(clips_j)
        for n, layer in enumerate(self.encoder.layers):
            if n == self.n_layers - 1:
                CLS_i_Q = layer.enc_self_attn.multihead_attn.obtain_Q(output_i[:, 0, :].unsqueeze(1))
                CLS_j_Q = layer.enc_self_attn.multihead_attn.obtain_Q(output_j[:, 0, :].unsqueeze(1))
                CLIP_j_K = layer.enc_self_attn.multihead_attn.obtain_K(output_j[:, 1:, :])
                break
            output_i = layer(output_i, self_attn_mask=None)
            output_j = layer(output_j, self_attn_mask=None)
        # Mask to generate intenstified samples
        atten_scores1 = torch.matmul(CLS_i_Q, CLIP_j_K.transpose(2,3)).squeeze()
        atten_scores2 = torch.matmul(CLS_j_Q, CLIP_j_K.transpose(2, 3)).squeeze()

        atten_scores = torch.softmax(atten_scores1 / self.scale_factor, dim=-1) + \
                       torch.softmax(atten_scores2 / self.scale_factor, dim=-1)

        atten_scores_j = torch.mean(atten_scores, dim=1)
        # [b_size x clips_num]


        _, indeices_j = torch.sort(atten_scores_j, dim=-1)
        masked_j = indeices_j[:, 0:self.mask_num].reshape(-1) + self.bias
        mask_j = torch.ones(b_size * self.clips_num, self.d_model).cuda()
        mask_j[masked_j] = 0
        mask_j = mask_j.reshape(b_size, self.clips_num, -1)


        #crucial_clips = crucial_clips.unsqueeze(2)

        # generate positive samples
        #crucial_mask = crucial_clips.repeat(1, 1, self.d_model)
        #crucial_j = crucial_mask * clips_j

        random_mask_i = torch.bernoulli(torch.full((b_size, self.clips_num + 1), 0.15)).unsqueeze(2)
        random_mask_j = torch.bernoulli(torch.full((b_size, self.clips_num + 1), 0.15)).unsqueeze(2)
        random_mask_i = random_mask_i.repeat(1, 1, self.d_model).cuda()
        random_mask_j = random_mask_j.repeat(1, 1, self.d_model).cuda()

        mask_strategy = torch.tensor([0.8, 0.1, 0.1])
        strategy_id = torch.multinomial(mask_strategy, 1)
        if strategy_id == 0:
            pred_i = clips_i * (1 - random_mask_i)
            pred_j = clips_j * (1 - random_mask_j)
        elif strategy_id == 1:
            idx = torch.randint(self.clips_num+1, (1,))
            replacing_i = clips_i[:, idx:idx+1, :].repeat(1, self.clips_num+1, 1)
            replacing_j = clips_j[:, idx:idx+1, :].repeat(1, self.clips_num+1, 1)
            pred_i = clips_i * (1 - random_mask_i) + random_mask_i * replacing_i
            pred_j = clips_j * (1 - random_mask_j) + random_mask_j * replacing_j
        elif strategy_id == 2:
            pred_i = clips_i
            pred_j = clips_j

        pred_i = self.pred_layer(self.trm(pred_i))
        pred_j = self.pred_layer(self.trm(pred_j))
        re_loss = self.huber(clips_i * random_mask_i, pred_i * random_mask_i) + \
                  self.huber(clips_j * random_mask_j, pred_j * random_mask_j)


        return   mask_j, re_loss / 2





