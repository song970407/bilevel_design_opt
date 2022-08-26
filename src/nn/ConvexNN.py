from typing import Union, List

import torch
import torch.nn as nn

from src.nn.NonnegativeLinear import NonnegativeLinear1, NonnegativeLinear2, NonnegativeLinear3


class PartialNN1(nn.Module):
    def __init__(self,
                 in_nic_features: int,
                 in_ic_features: int,
                 out_nic_features: int,
                 out_ic_features: int,
                 nic_activation: nn.Module = None,
                 ic_activation: nn.Module = None):
        super(PartialNN1, self).__init__()
        self.in_nic_features = in_nic_features
        self.in_ic_features = in_ic_features
        self.out_nic_features = out_nic_features
        self.out_ic_features = out_ic_features
        self.nic_nic_layer = nn.Linear(in_nic_features, out_nic_features)
        self.nic_activation = nic_activation
        self.nic_ic_layer = nn.Linear(in_nic_features, out_ic_features)
        self.ic_ic_layer = nn.Linear(in_features=in_ic_features, out_features=out_ic_features)
        self.ic_activation = ic_activation

    def forward(self, x):
        """
        :param x: torch.Tensor, B x (in_features[0]+in_features[1])
        :return: B x (out_features[0]+out_features[1])
        """
        nic_input, ic_input = torch.split(x, [self.in_nic_features, self.in_ic_features], dim=1)
        return torch.cat([self.nic_activation(self.nic_nic_layer(nic_input)), self.ic_activation(self.nic_ic_layer(nic_input) + self.ic_ic_layer(ic_input))], dim=1)



class PartialConvexNN1(nn.Module):
    def __init__(self,
                 in_nic_features: int,
                 in_ic_features: int,
                 out_nic_features: int,
                 out_ic_features: int,
                 bias: bool = True,
                 nic_activation: nn.Module = None,
                 ic_activation: nn.Module = None,
                 is_end: bool = False):
        """
        Args:
            in_nic_features:
            in_ic_features:
            out_nic_features:
            out_ic_features:
            bias:
            nic_activation:
            ic_activation:
            is_end:
        """
        super(PartialConvexNN1, self).__init__()
        self.in_nic_features = in_nic_features
        self.in_ic_features = in_ic_features
        self.out_nic_features = out_nic_features
        self.out_ic_features = out_ic_features
        if not is_end:
            self.nic_nic_layer = nn.Linear(in_nic_features, out_nic_features)
            self.nic_activation = nic_activation
        self.nic_ic_layer = nn.Linear(in_nic_features, out_ic_features)
        self.ic_ic_layer = NonnegativeLinear1(in_features=in_ic_features,
                                              out_features=out_ic_features,
                                              bias=bias)
        self.ic_activation = ic_activation
        self.is_end = is_end

    def forward(self, x):
        """
        :param x: torch.Tensor, B x (in_features[0]+in_features[1])
        :return: B x (out_features[0]+out_features[1])
        """
        nic_input, ic_input = torch.split(x, [self.in_nic_features, self.in_ic_features], dim=1)
        if self.is_end:
            return self.ic_activation(self.nic_ic_layer(nic_input) + self.ic_ic_layer(ic_input))
        else:
            return torch.cat([self.nic_activation(self.nic_nic_layer(nic_input)),
                              self.ic_activation(self.nic_ic_layer(nic_input) + self.ic_ic_layer(ic_input))], dim=1)


class PartialConvexNN2(PartialConvexNN1):
    def __init__(self,
                 in_nic_features: int,
                 in_ic_features: int,
                 out_nic_features: int,
                 out_ic_features: int,
                 bias: bool = True,
                 nic_activation: nn.Module = None,
                 ic_activation: nn.Module = None,
                 is_end: bool = False):
        super(PartialConvexNN2, self).__init__(in_nic_features,
                                               in_ic_features,
                                               out_nic_features,
                                               out_ic_features,
                                               bias,
                                               nic_activation,
                                               ic_activation,
                                               is_end)
        self.ic_ic_layer = NonnegativeLinear2(in_features=in_ic_features,
                                              out_features=out_ic_features,
                                              bias=bias)


class PartialConvexNN3(PartialConvexNN1):
    def __init__(self,
                 in_nic_features: int,
                 in_ic_features: int,
                 out_nic_features: int,
                 out_ic_features: int,
                 bias: bool = True,
                 nic_activation: nn.Module = None,
                 ic_activation: nn.Module = None,
                 is_end: bool = False):
        super(PartialConvexNN3, self).__init__(in_nic_features,
                                               in_ic_features,
                                               out_nic_features,
                                               out_ic_features,
                                               bias,
                                               nic_activation,
                                               ic_activation,
                                               is_end)
        self.ic_ic_layer = NonnegativeLinear3(in_features=in_ic_features,
                                              out_features=out_ic_features,
                                              bias=bias)
