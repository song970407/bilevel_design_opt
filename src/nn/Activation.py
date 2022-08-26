import torch
import torch.nn as nn


class LearnableLeakyReLU(nn.Module):
    def __init__(self,
                 input_dim: int,
                 negative_slope: float = 1.0):
        """
        :param input_dim: int, dimension of input
        :param negative_slope: float, slope of negative part
        """
        super(LearnableLeakyReLU, self).__init__()
        bound = torch.empty(size=(1, input_dim))
        nn.init.uniform_(bound, -1.0, 1.0)
        self.bound = torch.nn.Parameter(bound)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        :param x: torch.tensor, [, self.x_dim]
        :return:
        """
        return torch.clamp(x-self.bound, min=0.0) + self.negative_slope * torch.clamp(x-self.bound, max=0.0)


class ConvexPReLU1(nn.Module):
    """
    Guarantee convexity by using Clipping Method
    """
    def __init__(self,
                 input_dim: int,
                 is_convex: bool = True):
        """
        input_dim: int
        :param is_convex: bool
        """
        super(ConvexPReLU1, self).__init__()
        a_value = torch.empty(size=(input_dim, ))
        if is_convex:
            nn.init.constant_(a_value, 0.5)
        else:
            nn.init.constant_(a_value, 1.5)
        self.a = torch.nn.Parameter(a_value)
        self.is_convex = is_convex

    def project_params(self):
        if self.is_convex:
            self.a.data = self.a.data.clamp(min=0.0, max=1.0)
        else:
            self.a.data = self.a.data.clamp(min=1.0)

    def forward(self, x):
        return torch.clamp(x, min=0.0) + self.a * torch.clamp(x, max=0.0)
