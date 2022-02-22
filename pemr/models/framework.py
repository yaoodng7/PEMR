import torch.nn as nn


class Framework(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(Framework, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        # We use a MLP as projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i, h_pos, h_neg, loss_pred = self.encoder.trm_mask(x_i, x_j)
        z_i = self.projector(h_i)
        z_pos = self.projector(h_pos)
        z_neg = self.projector(h_neg)
        return  z_i, z_pos, z_neg, loss_pred