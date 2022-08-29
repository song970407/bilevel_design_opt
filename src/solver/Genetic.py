import torch
import torch.nn as nn


class GeneticSolver(nn.Module):
    def __init__(self, solver_config):
        super(GeneticSolver, self).__init__()
