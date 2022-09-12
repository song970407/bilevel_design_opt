from time import perf_counter

import numpy as np
from cmaes import CMA
import torch
import torch.nn as nn
import torch.optim as th_opt
import dgl

from src.utils.get_graph import generate_full_graph


class LowerLevelOptimizer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 num_states: int,
                 num_actions: int,
                 history_len: int,
                 ridge_coefficient: float,
                 smoothness_coefficient: float,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 max_iter: int = 200,
                 loss_threshold: float = 1e-8,
                 is_logging: bool = True,
                 device: str = 'cpu',
                 opt_name: str = 'Adam',
                 opt_config: dict = None):
        """
        :param model: (nn.Module) nn-parametrized dynamic model
        :param graph: (dgl.graph) graph-structure
        :param num_states: (int) number of states
        :param num_actions: (int) number of actions
        :param ridge_coefficient: (float) coefficient for Ridge regularizer
        :param smoothness_coefficient: (float) coefficient for smoothness regularization
        :param u_min: (float) the lower bound of action values
        :param u_max: (float) the upper bound of action values
        :param max_iter: (int) the maximum iteration for MPC optimization problem
        :param is_logging:
        :param device: (str) The computation device that is used for the MPC optimization
        :param opt_config: (dict)
        :param scheduler_config: (dict)
        """
        super(LowerLevelOptimizer, self).__init__()

        self.model = model.to(device)
        self.graph = graph.to(device)
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_dim = 1
        self.action_dim = 1
        self.history_len = history_len
        self.ridge_coefficient = ridge_coefficient
        self.smoothness_coefficient = smoothness_coefficient
        self.u_min = u_min
        self.u_max = u_max
        self.max_iter = max_iter
        self.loss_threshold = loss_threshold
        self.is_logging = is_logging
        self.opt_name = opt_name
        self.opt_config = opt_config

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print("Initializing solver with {}".format(device))
        self.device = device
        # Initialize Control Variable
        self.crit = torch.nn.MSELoss(reduction='none')
        self.criteria_smoothness = torch.nn.MSELoss(reduction='none')

    def set_position(self, position):
        with torch.no_grad():
            state_pos = self.graph.nodes['state'].data['pos'][:self.num_states]
            graph_list = []
            for i in range(self.num_targets):
                graph_list.append(generate_full_graph(state_pos, position, self.device))
            self.graph = dgl.batch(graph_list)

    def compute_individual_loss(self, us):
        self.hist_xs = torch.zeros((self.total_states, self.history_len, 1)).to(self.device)
        self.hist_us = torch.zeros((self.total_actions, self.history_len - 1, 1)).to(self.device)
        pred_xs = self.predict_future(us)
        loss_objective = self.crit(pred_xs, self.target).mean(dim=(1, 2)).split(self.num_states_list)
        loss_ridge = (self.ridge_coefficient * torch.mean(us ** 2, dim=(1, 2))).split(self.num_actions_list)
        prev_us = torch.cat([self.hist_us[:, -1:], us[:, :-1]], dim=1)
        loss_smoothness = (self.smoothness_coefficient * self.criteria_smoothness(us, prev_us).mean(dim=(1, 2))).split(
            self.num_actions_list)
        loss_list = []
        for i in range(len(loss_objective)):
            loss_list.append(torch.sum(loss_objective[i]) + torch.sum(loss_ridge[i]) + torch.sum(loss_smoothness[i]))
        return torch.stack(loss_list)

    def compute_total_loss(self, us):
        return torch.mean(self.compute_individual_loss(us))

    def compute_lambda(self, optimal_u):
        jacobian_u = torch.autograd.functional.jacobian(self.compute_total_loss, optimal_u)
        optimal_lambda_min = (optimal_u == self.u_min).int() * jacobian_u
        optimal_lambda_max = - (optimal_u == self.u_max).int() * jacobian_u
        return optimal_lambda_min, optimal_lambda_max

    @staticmethod
    def record_log(target, individual_loss, total_loss, time, total_runtime, best_idx_list, best_us, best_loss):
        log = {
            'target_xs': target,
            'individual_loss': individual_loss,
            'total_loss': total_loss,
            'time': time,
            'total_runtime': total_runtime,
            'best_idx': best_idx_list,
            'best_us': best_us,
            'best_loss': best_loss
        }
        return log

    def solve(self, position, target_list):
        """
        :param position: torch.Tensor
        :return:
        """
        num_targets = len(target_list)
        self.num_targets = num_targets
        self.target = torch.cat(target_list, dim=0).to(self.device)
        self.num_states_list = [self.num_states for _ in range(num_targets)]
        self.num_actions_list = [self.num_actions for _ in range(num_targets)]
        self.set_position(position)
        self.total_states = self.graph.number_of_nodes('state')
        self.total_actions = self.graph.number_of_nodes('action')
        self.receding_horizon = self.target.shape[1]

        loop_start = perf_counter()
        with torch.no_grad():
            us = torch.empty(size=(self.total_actions, self.receding_horizon, self.action_dim)).to(self.device)
            nn.init.constant_(us, (self.u_max - self.u_min) / 2)
            us = torch.nn.Parameter(us).to(self.device)
        opt = th_opt.Adam([us], **self.opt_config)
        opt = getattr(th_opt, self.opt_name)([us], **self.opt_config)

        individual_loss_trajectory = []
        total_loss_trajectory = []  # the loss over the optimization steps
        us_trajectory = []  # the optimized actions over the optimization steps
        time_trajectory = []  # the optimization runtime over the optimization steps
        for i in range(self.max_iter):
            iter_start = perf_counter()
            individual_loss = self.compute_individual_loss(us)
            loss = torch.mean(individual_loss)
            individual_loss_trajectory.append(individual_loss.cpu().detach().numpy())
            total_loss_trajectory.append(loss.item())
            us_trajectory.append(us.data.cpu().detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            time_trajectory.append(perf_counter() - iter_start)
        with torch.no_grad():
            iter_start = perf_counter()
            individual_loss = self.compute_individual_loss(us)
            loss = torch.mean(individual_loss)
            individual_loss_trajectory.append(individual_loss.cpu().detach().numpy())
            total_loss_trajectory.append(loss.item())
            us_trajectory.append(us.data.cpu().detach())
            time_trajectory.append(perf_counter() - iter_start)
        # Return the best us
        individual_loss_trajectory = np.stack(individual_loss_trajectory, axis=0)
        total_loss_trajectory = np.stack(total_loss_trajectory)
        idx_list = np.argmin(individual_loss_trajectory, axis=0)
        optimal_u = []
        curr_action_idx = 0
        best_loss = 0.0
        for (i, idx) in enumerate(idx_list):
            optimal_u.append(us_trajectory[idx][curr_action_idx:curr_action_idx + self.num_actions_list[i]])
            curr_action_idx += self.num_actions_list[i]
            best_loss += individual_loss_trajectory[idx, i]
        optimal_u = torch.cat(optimal_u, dim=0).to(self.device)
        optimal_lambda_min, optimal_lambda_max = self.compute_lambda(optimal_u)
        if self.is_logging:
            log = self.record_log(target=self.target.cpu().detach().numpy(),
                                  individual_loss=individual_loss_trajectory,
                                  total_loss=total_loss_trajectory,
                                  time=np.array(time_trajectory),
                                  total_runtime=perf_counter() - loop_start,
                                  best_idx_list=idx_list,
                                  best_us=optimal_u.cpu().detach().numpy(),
                                  best_loss=best_loss)
        else:
            log = None
        return optimal_u, optimal_lambda_min, optimal_lambda_max, log

    def predict_future(self, us):
        return self.model.multistep_prediction(self.graph, self.hist_xs, self.hist_us, us)


class CMAESSolver(nn.Module):
    def __init__(self, solver_config):
        super(CMAESSolver, self).__init__()
        self.history_len = solver_config['history_len']
        self.num_samples = solver_config['num_samples']
        self.upper_max_iter = solver_config['upper_max_iter']
        self.lower_max_iter = solver_config['lower_max_iter']
        self.lower_opt_name = solver_config['lower_opt_name']
        self.lower_opt_config = solver_config['lower_opt_config']
        self.initial_mean = solver_config['initial_mean']
        self.initial_std = solver_config['initial_std']
        self.model = solver_config['model']
        self.upper_bound = solver_config['upper_bound']
        self.lower_bound = solver_config['lower_bound']
        self.device = solver_config['device']

    def solve(self, target_list, state_pos, action_pos):
        torch_state_pos = torch.from_numpy(state_pos).float()
        torch_action_pos = torch.from_numpy(action_pos).float()
        graph = generate_full_graph(torch_state_pos, torch_action_pos)
        lower_level_optimizer = LowerLevelOptimizer(model=self.model,
                                                    graph=graph,
                                                    num_states=state_pos.shape[0],
                                                    num_actions=action_pos.shape[0],
                                                    ridge_coefficient=0.0,
                                                    smoothness_coefficient=0.0,
                                                    history_len=self.history_len,
                                                    u_min=self.lower_bound[0],
                                                    u_max=self.lower_bound[1],
                                                    max_iter=self.lower_max_iter,
                                                    device=self.device,
                                                    opt_name=self.lower_opt_name,
                                                    opt_config=self.lower_opt_config)
        optimizer = CMA(mean=np.ones(action_pos.size) * self.initial_mean, sigma=self.initial_std)
        total_loss_trajectory = []
        position_trajectory = []
        us_trajectory = []
        lower_level_log_trajectory = []
        best_action_pos = None
        best_fitness = float('inf')
        best_idx = None
        best_us = None
        for upper_itr in range(self.upper_max_iter):
            print('Now Upper Iteration: [{}] / [{}]'.format(upper_itr + 1, self.upper_max_iter))
            solutions = []
            loss_traj = []
            pos_traj = []
            u_traj = []
            log_traj = []
            for i in range(self.num_samples):
                print('Now Sample: [{}] / [{}]'.format(i, self.num_samples))
                x = optimizer.ask()
                sampled_action_pos = np.clip(x.reshape(action_pos.shape), a_min=self.upper_bound[0],
                                             a_max=self.upper_bound[1])
                sampled_action_pos = torch.from_numpy(sampled_action_pos).float().to(self.device)
                optimal_us, _, _, log = lower_level_optimizer.solve(sampled_action_pos, target_list)
                fitness = log['best_loss']
                with torch.no_grad():
                    loss_traj.append(log['best_loss'])
                    pos_traj.append(sampled_action_pos.cpu().detach().numpy())
                    u_traj.append(optimal_us.cpu().detach().numpy())
                    log_traj.append(log)
                if fitness < best_fitness:
                    print('Best Design Variable Found, Loss: {}'.format(fitness))
                    best_action_pos = sampled_action_pos
                    best_fitness = fitness
                    best_idx = upper_itr
                    best_us = optimal_us.cpu().detach().numpy()
                solutions.append((x, fitness))
            optimizer.tell(solutions)
            with torch.no_grad():
                total_loss_trajectory.append(log_traj)
                position_trajectory.append(pos_traj)
                us_trajectory.append(u_traj)
                lower_level_log_trajectory.append(log_traj)
        total_loss_trajectory = np.array(total_loss_trajectory)
        position_trajectory = np.array(position_trajectory)
        us_trajectory = np.array(us_trajectory)

        opt_log = {
            'total_loss_trajectory': total_loss_trajectory,
            'position_trajectory': position_trajectory,
            'us_trajectory': us_trajectory,
            'lower_level_log_trajectory': lower_level_log_trajectory,
            'best_idx': best_idx,
            'best_loss': best_fitness,
            'best_position': best_action_pos,
            'best_us': best_us
        }
        return best_action_pos, opt_log
