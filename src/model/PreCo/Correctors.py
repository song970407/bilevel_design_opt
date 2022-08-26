import torch
import torch.nn as nn

from src.nn.mpnn import MPNN


class Corrector(nn.Module):

    def __init__(self, x_dim, h_dim, mlp_h_dim=16):
        super(Corrector, self).__init__()
        self.h2h = nn.Sequential(
            nn.Linear(h_dim + x_dim + 2, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim)
        )

    def forward(self, g, h, x):
        with g.local_scope():
            inp = torch.cat([h, x, g.ndata['pos']], dim=-1)
            return self.h2h(inp)


class HeteroCorrector(nn.Module):
    def __init__(self, x_dim, h_dim, mlp_h_dim=64):
        super(HeteroCorrector, self).__init__()
        self.h2h = nn.Sequential(
            nn.Linear(h_dim + x_dim + 2, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim)
        )

    def forward(self, g, h, x):
        with g.local_scope():
            inp = torch.cat([h, x, g.nodes['state'].data['pos']], dim=-1)
            return self.h2h(inp)


class HeteroCorrector2(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(HeteroCorrector2, self).__init__()
        self.h2h = MPNN(node_indim=x_dim + h_dim + 2,
                        node_outdim=h_dim,
                        edge_outdim=h_dim)

    def forward(self, g, h, x):
        with g.local_scope():
            inp = torch.cat([h, x, g.nodes['state'].data['pos']], dim=-1)
            uh, _ = self.h2h(g[('state', 's2s', 'state')], inp)
            return uh


class LinearPreCOCorrector(Corrector):

    def __init__(self, x_dim, h_dim):
        super(LinearPreCOCorrector, self).__init__(x_dim, h_dim)
        self.h2h = nn.Linear(h_dim + x_dim + 2, h_dim)
