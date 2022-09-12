import torch
import torch.nn as nn


class GeneticSolver(nn.Module):
    def __init__(self, solver_config):
        super(GeneticSolver, self).__init__()
        self.max_iter = solver_config['max_iter']
        self.num_parents = solver_config['num_parents']

        self.model = solver_config['model']
        self.upper_bound = solver_config['upper_bound']
        self.lower_bound = solver_config['lower_bound']
        self.device = solver_config['device']

    def solve(self, target, state_pos, action_pos):
        return
