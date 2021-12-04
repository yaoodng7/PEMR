import torch
import torch.nn as nn




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BTLoss(nn.Module):
    def __init__(self, batch_size, lambd1, lambd2, projection_dim):
        super(BTLoss, self).__init__()
        self.batch_size = batch_size
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(projection_dim, affine=True)

    def forward(self, z_i, z_j, z_neg):
        # empirical cross-correlation matrix
        c_cru = self.bn(z_i).T @ self.bn(z_j)
        c_imma = self.bn(z_i).T @ self.bn(z_neg)
        c_cru.div_(self.batch_size)
        c_imma.div_(self.batch_size)
        # torch.distributed.all_reduce(c)
        cru_on_diag = torch.diagonal(c_cru).add_(-1).pow_(2).sum()
        imma_on_diag = torch.diagonal(c_imma).pow_(2).sum()
        cru_off_diag = off_diagonal(c_cru).pow_(2).sum()
        loss = cru_on_diag + self.lambd1 * imma_on_diag + self.lambd2 * cru_off_diag
        return loss
