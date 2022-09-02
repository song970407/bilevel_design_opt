import torch.nn as nn


class DecoderNN(nn.Module):
    def __init__(self, hidden_dim, output_dim=1):
        super(DecoderNN, self).__init__()
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
