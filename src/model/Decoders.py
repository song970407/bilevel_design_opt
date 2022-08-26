import torch.nn as nn

from src.nn.NonnegativeLinear import NonnegativeLinear1
from src.nn.Activation import LearnableLeakyReLU, ConvexPReLU1


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.decoder(x)


class ConvexDecoder(Decoder):
    def __init__(self, hidden_dim, is_convex, output_dim=1):
        super(ConvexDecoder, self).__init__(hidden_dim, output_dim)
        self.decoder = nn.Sequential(
            NonnegativeLinear1(hidden_dim, 64),
            ConvexPReLU1(64, is_convex=is_convex),
            NonnegativeLinear1(64, 32),
            ConvexPReLU1(32, is_convex=is_convex),
            NonnegativeLinear1(32, 16),
            ConvexPReLU1(16, is_convex=is_convex),
            NonnegativeLinear1(16, output_dim)
        )
