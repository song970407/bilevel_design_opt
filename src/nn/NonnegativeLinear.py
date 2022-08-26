import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter


class NonnegativeLinear1(nn.Module):
    """
    Guarantee Non-negativity by using Clipping Method
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(NonnegativeLinear1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.positive_weight = Parameter(torch.Tensor(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
        # self.bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.positive_weight, a=0.0, b=2 / (self.in_features + self.out_features))
        # self.positive_weight.data = self.positive_weight.data.abs()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.positive_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def project_params(self):
        self.positive_weight.data = self.positive_weight.data.clamp(min=0.0)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.positive_weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class NonnegativeLinear2(nn.Module):
    """
    Guarantee Non-negativity by using ReLU Reparameterization Trick
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(NonnegativeLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
        # self.bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, a=0.0, b=2 / (self.in_features + self.out_features))
        # self.positive_weight.data = self.positive_weight.data.abs()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, F.relu(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NonnegativeLinear3(nn.Module):
    """
    Guarantee Non-negativity by using Absolute Reparameterization Trick
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(NonnegativeLinear3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
        # self.bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, a=- 2 / (self.in_features + self.out_features), b=2 / (self.in_features + self.out_features))
        # self.positive_weight.data = self.positive_weight.data.abs()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, torch.abs(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
