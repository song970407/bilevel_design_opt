import torch
import torch.nn as nn


class SingleLayerSolver(nn.Module):
    def __init__(self, solver_config):
        super(SingleLayerSolver, self).__init__()
