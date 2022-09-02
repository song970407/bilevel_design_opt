import torch.nn as nn

from src.nn.NonnegativeLinear import NonnegativeLinear1
from src.nn.Activation import LearnableLeakyReLU, ConvexPReLU1


class DecoderConvexNN(nn.Module):
    def __init__(self, hidden_dim, is_convex, output_dim=1):
        super(DecoderConvexNN, self).__init__()
        self.decoder = nn.Sequential(
            NonnegativeLinear1(hidden_dim, 64),
            ConvexPReLU1(64, is_convex=is_convex),
            NonnegativeLinear1(64, 32),
            ConvexPReLU1(32, is_convex=is_convex),
            NonnegativeLinear1(32, 16),
            ConvexPReLU1(16, is_convex=is_convex),
            NonnegativeLinear1(16, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)
