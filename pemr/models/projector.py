import torch.nn as nn
import torchvision


class PrSimCLR(nn.Module):
    def __init__(self, projection_dim, n_features):
        super(PrSimCLR, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, h_i):
        z_i = self.projector(h_i)   # (b, 256)
        return z_i