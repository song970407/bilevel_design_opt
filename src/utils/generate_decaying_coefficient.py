import torch

def generate_decaying_coefficient(decaying_gamma, receding_horizon, device):
    decaying_factor = []
    for i in range(receding_horizon):
        decaying_factor.append(decaying_gamma ** i)
    return torch.tensor(decaying_gamma, device=device)