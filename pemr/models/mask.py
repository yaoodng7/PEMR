import torch
import torch.nn as nn
import numpy as np
from .transformer_encoder import Encoder
class Mask(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, n_heads, clips_num, bias, mask_num):
        super(Mask, self).__init__()
        self.encoder = Encoder(n_layers, d_k, d_v, d_model, n_heads)
        self.n_layers = n_layers
        self.clips_num = clips_num
        self.d_model = d_model
        self.scale_factor = np.sqrt(d_model)
        self.threshold = 0
        self.bias = bias
        self.mask_num = mask_num

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

        # ----------------------------------
        # Mask to generate augmented samples
        # ----------------------------------
        atten_scores1 = torch.matmul(CLS_i_Q, CLIP_j_K.transpose(2,3)).squeeze()
        atten_scores2 = torch.matmul(CLS_j_Q, CLIP_j_K.transpose(2, 3)).squeeze()

        atten_scores = torch.softmax(atten_scores1 / self.scale_factor, dim=-1) + \
                       torch.softmax(atten_scores2 / self.scale_factor, dim=-1)

        atten_scores = torch.mean(atten_scores, dim=1)
        # [b_size x clips_num]

        self.bias = self.bias.cuda()
        _, indeices_j = torch.sort(atten_scores, dim=-1)
        masked_j = indeices_j[:, 0:self.mask_num].reshape(-1) + self.bias
        mask_j = torch.ones(b_size * self.clips_num, self.d_model).cuda()
        mask_j[masked_j] = 0
        mask_j = mask_j.reshape(b_size, self.clips_num, -1)
        return   mask_j





