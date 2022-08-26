import torch
import torch.nn as nn

from src.nn.NonnegativeLinear import NonnegativeLinear1
from src.nn.Activation import ConvexPReLU1

def get_positive_weights(module: nn.Module):
    param_list = []
    for n, p in module.named_parameters():
        if 'positive_weight' in n:
            param_list.append(p)
    return param_list


def compute_linear_penalty(params, coeff: float = 1.0, eps=1e-5):
    mask = params < 0.0  # find negative weights
    num_non_positive = mask.sum()
    cost = coeff * (params.abs() * mask).sum() / (num_non_positive + eps)
    return cost


def get_nonnegative_penalty(module: nn.Module, penalty_func=compute_linear_penalty, device='cpu'):
    pos_ws = get_positive_weights(module)

    penalty = torch.zeros(1).to(device)
    for pos_w in pos_ws:
        penalty += penalty_func(pos_w)
    return penalty


def project_params(module: nn.Module):
    for m in module.modules():
        if isinstance(m, NonnegativeLinear1):
            m.project_params()
        elif isinstance(m, ConvexPReLU1):
            m.project_params()
