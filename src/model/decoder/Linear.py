import torch.nn as nn

from src.nn.NonnegativeLinear import NonnegativeLinear1
from src.nn.Activation import LearnableLeakyReLU, ConvexPReLU1


class DecoderLinear(nn.Module):
    def __init__(self, hidden_dim, output_dim=1):
        super(DecoderLinear, self).__init__()
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.decoder(x)