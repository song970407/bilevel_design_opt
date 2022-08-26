import time
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as th_opt
import dgl


def get_discount_factor(horizon_length, gamma):
    g = 1.0
    gs = [g]
    for i in range(1, horizon_length):
        g *= gamma
        gs.append(g)
    return torch.tensor(gs)


def get_default_opt_config():
    opt_config = {'lr': 1e-2}
    return opt_config


def get_default_scheduler_config():
    scheduler_config = {'name': 'ReduceLROnPlateau', 'patience': 2, 'factor': 0.5, 'min_lr': 1e-3}
    return scheduler_config


class MPC(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 num_states: int,
                 num_actions: int,
                 num_states_list: list,
                 num_actions_list: list,
                 state_dim: int,
                 action_dim: int,
                 ridge_coefficient: float,
                 smoothness_coefficient: float,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 max_iter: int = 200,
                 loss_threshold: float = 1e-8,
                 is_logging: bool = True,
                 device: str = 'cpu',
                 opt_config: dict = None,
                 scheduler_config: dict = None):
        """
        :param model: (nn.Module) nn-parametrized dynamic model
        :param graph: (dgl.graph) graph-structure
        :param num_states: (int) number of states
        :param num_actions: (int) number of actions
        :param state_dim: (int) dimension of each state, usually 1
        :param action_dim: (int) dimension of each action, usually 1
        :param receding_horizon: (int) receding horizon
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
        super(MPC, self).__init__()

        self.model = model.to(device)
        self.graph = graph.to(device)
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_states_list = num_states_list
        self.num_actions_list = num_actions_list
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ridge_coefficient = ridge_coefficient
        self.smoothness_coefficient = smoothness_coefficient
        self.u_min = u_min
        self.u_max = u_max
        self.max_iter = max_iter
        self.loss_threshold = loss_threshold
        self.is_logging = is_logging

        opt_config = get_default_opt_config() if opt_config is None else opt_config
        self.opt_config = opt_config

        scheduler_config = get_default_scheduler_config() if scheduler_config is None else scheduler_config
        self.scheduler_config = scheduler_config

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Initializing solver with {}".format(device))
        self.device = device
        # Initialize Control Variable
        self.crit = torch.nn.MSELoss(reduction='none')
        self.criteria_smoothness = torch.nn.MSELoss(reduction='none')

    def compute_loss(self, us, hist_xs, hist_us, target_xs):
        receding_horizon = target_xs.shape[1]
        gs = get_discount_factor(receding_horizon, 0.8).reshape(1, -1, 1).to(self.device)
        pred_xs = self.predict_future(hist_xs, hist_us, us)
        loss_objective = self.crit(gs * pred_xs, gs * target_xs).mean(dim=(1, 2)).split(self.num_states_list)
        loss_ridge = (self.ridge_coefficient * torch.mean(us ** 2, dim=(1, 2))).split(self.num_actions_list)
        prev_us = torch.cat([hist_us[:, -1:], us[:, :-1]], dim=1)
        loss_smoothness = (self.smoothness_coefficient * self.criteria_smoothness(us, prev_us).mean(dim=(1, 2))).split(
            self.num_actions_list)
        loss_list = []
        for i in range(len(loss_objective)):
            loss_list.append(torch.sum(loss_objective[i]) + torch.sum(loss_ridge[i]) + torch.sum(loss_smoothness[i]))
        return torch.stack(loss_list)

    @staticmethod
    def record_log(hist_xs, hist_us, target_xs, individual_loss, total_loss, time, total_runtime, best_idx_list, best_us):
        log = {
            'hist_xs': hist_xs,
            'hist_us': hist_us,
            'target_xs': target_xs,
            'individual_loss': individual_loss,
            'total_loss': total_loss,
            'time': time,
            'total_runtime': total_runtime,
            'best_idx': best_idx_list,
            'best_us': best_us
        }
        return log

    def solve(self, hist_xs, hist_us, target_xs):
        """
        :param hist_xs: [num_states x state_order x state dim]
        :param hist_us: [num_actions x action_order-1 x action dim]
        :param target_xs: [num_states x receding_horizon x state dim]
        :return:
        """
        loop_start = perf_counter()
        receding_horizon = target_xs.shape[1]
        with torch.no_grad():
            us = torch.empty(size=(self.num_actions, receding_horizon, self.action_dim)).to(self.device)
            nn.init.constant_(us, (self.u_max - self.u_min) / 2)
            us = torch.nn.Parameter(us).to(self.device)
        opt = th_opt.Adam([us], **self.opt_config)
        scheduler = th_opt.lr_scheduler.ReduceLROnPlateau(opt, **self.scheduler_config)
        individual_loss_trajectory = []
        total_loss_trajectory = []  # the loss over the optimization steps
        us_trajectory = []  # the optimized actions over the optimization steps
        time_trajectory = []  # the optimization runtime over the optimization steps
        for i in range(self.max_iter):
            iter_start = perf_counter()
            individual_loss = self.compute_loss(us, hist_xs, hist_us, target_xs)
            loss = torch.sum(individual_loss)
            individual_loss_trajectory.append(individual_loss.cpu().detach().numpy())
            total_loss_trajectory.append(loss.item())
            us_trajectory.append(us.data.cpu().detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step(loss)
            with torch.no_grad():
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            time_trajectory.append(perf_counter() - iter_start)
        with torch.no_grad():
            iter_start = perf_counter()
            individual_loss = self.compute_loss(us, hist_xs, hist_us, target_xs)
            loss = torch.sum(individual_loss)
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
        for (i, idx) in enumerate(idx_list):
            optimal_u.append(us_trajectory[idx][curr_action_idx:curr_action_idx + self.num_actions_list[i]])
            curr_action_idx += self.num_actions_list[i]
        optimal_u = torch.cat(optimal_u, dim=0).to(self.device)
        if self.is_logging:
            log = self.record_log(hist_xs=hist_xs.cpu().numpy(),
                                  hist_us=hist_us.cpu().numpy(),
                                  target_xs=target_xs.cpu().numpy(),
                                  individual_loss=individual_loss_trajectory,
                                  total_loss=total_loss_trajectory,
                                  time=np.array(time_trajectory),
                                  total_runtime=perf_counter() - loop_start,
                                  best_idx_list=idx_list,
                                  best_us=optimal_u.cpu().detach().numpy()
                                  )
        else:
            log = None
        return optimal_u, log

    def predict_future(self, hist_xs, hist_us, us):
        h0 = self.model.filter_history(self.graph, hist_xs, hist_us)
        prediction = self.model.multi_step_prediction(self.graph, h0, us)
        return prediction

    def predict_multiple_future(self, hist_xs, hist_us, us):
        h0 = self.model.filter_history(self.multiple_graph, hist_xs, hist_us)
        prediction = self.model.multi_step_prediction(self.multiple_graph, h0, us)
        return prediction


class MPCParallel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 num_states: int,
                 num_actions: int,
                 num_states_list: list,
                 num_actions_list: list,
                 state_dim: int,
                 action_dim: int,
                 receding_horizon: int,
                 ridge_coefficient: float,
                 smoothness_coefficient: float,
                 u_min: float = 0.0,
                 u_max: float = 1.0,
                 max_iter: int = 200,
                 loss_threshold: float = 1e-8,
                 is_logging: bool = True,
                 device: str = 'cpu',
                 opt_config: dict = None,
                 scheduler_config: dict = None):
        """
        :param model: (nn.Module) nn-parametrized dynamic model
        :param graph: (dgl.graph) graph-structure
        :param num_states: (int) number of states
        :param num_actions: (int) number of actions
        :param state_dim: (int) dimension of each state, usually 1
        :param action_dim: (int) dimension of each action, usually 1
        :param receding_horizon: (int) receding horizon
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
        super(MPCParallel, self).__init__()

        self.model = model.to(device)
        self.graph = graph.to(device)
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_states_list = num_states_list
        self.num_actions_list = num_actions_list
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.receding_horizon = receding_horizon
        self.ridge_coefficient = ridge_coefficient
        self.smoothness_coefficient = smoothness_coefficient
        self.u_min = u_min
        self.u_max = u_max
        self.max_iter = max_iter
        self.loss_threshold = loss_threshold
        self.is_logging = is_logging

        opt_config = get_default_opt_config() if opt_config is None else opt_config
        self.opt_config = opt_config

        scheduler_config = get_default_scheduler_config() if scheduler_config is None else scheduler_config
        self.scheduler_config = scheduler_config

        if device is None:
            print("Running device is not given. Infer the running device from the system configuration ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Initializing solver with {}".format(device))
        self.device = device
        # Initialize Control Variable
        self.us = torch.empty(size=(self.num_actions, self.receding_horizon, self.action_dim)).to(self.device)
        nn.init.constant_(self.us, (u_max - u_min) / 2)
        self.crit = torch.nn.MSELoss(reduction='none')
        self.criteria_smoothness = torch.nn.MSELoss(reduction='none')

    def compute_loss(self, us, hist_xs, hist_us, target_xs):
        pred_xs = self.predict_future(hist_xs, hist_us, us)
        loss_objective = self.crit(pred_xs, target_xs).mean(dim=(1, 2)).split(self.num_states_list)
        loss_ridge = (self.ridge_coefficient * torch.mean(us ** 2, dim=(1, 2))).split(self.num_actions_list)
        prev_us = torch.cat([hist_us[:, -1:], us[:, :-1]], dim=1)
        loss_smoothness = (self.smoothness_coefficient * self.criteria_smoothness(us, prev_us).mean(dim=(1, 2))).split(
            self.num_actions_list)
        loss_list = []
        for i in range(len(loss_objective)):
            loss_list.append(torch.sum(loss_objective[i]) + torch.sum(loss_ridge[i]) + torch.sum(loss_smoothness[i]))
        return torch.stack(loss_list)

    @staticmethod
    def record_log(hist_xs, hist_us, target_xs, individual_loss, total_loss, time, total_runtime, best_idx_list, best_us):
        log = {
            'hist_xs': hist_xs,
            'hist_us': hist_us,
            'target_xs': target_xs,
            'individual_loss': individual_loss,
            'total_loss': total_loss,
            'time': time,
            'total_runtime': total_runtime,
            'best_idx': best_idx_list,
            'best_us': best_us
        }
        return log

    def solve(self, hist_xs, hist_us, target_xs):
        """
        :param hist_xs: [num_states x state_order x state dim]
        :param hist_us: [num_actions x action_order-1 x action dim]
        :param target_xs: [num_states x receding_horizon x state dim]
        :return:
        """
        loop_start = perf_counter()
        receding_horizon = target_xs.shape[1]
        with torch.no_grad():
            us = torch.nn.Parameter(self.us[:, -receding_horizon:]).to(self.device)
        opt = th_opt.Adam([us], **self.opt_config)
        scheduler = th_opt.lr_scheduler.ReduceLROnPlateau(opt, **self.scheduler_config)

        individual_loss_trajectory = []
        total_loss_trajectory = []  # the loss over the optimization steps
        us_trajectory = []  # the optimized actions over the optimization steps
        time_trajectory = []  # the optimization runtime over the optimization steps
        prev_loss = float('inf')
        max_patients = 10
        num_patients = 0
        for i in range(self.max_iter):
            iter_start = perf_counter()
            individual_loss = self.compute_loss(us, hist_xs, hist_us, target_xs)
            loss = torch.sum(individual_loss)
            individual_loss_trajectory.append(individual_loss.cpu().detach().numpy())
            total_loss_trajectory.append(loss.item())
            us_trajectory.append(us.data.cpu().detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step(loss)
            with torch.no_grad():
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            time_trajectory.append(perf_counter() - iter_start)
            if 0 <= prev_loss - loss.item() < self.loss_threshold:
                num_patients += 1
                if max_patients < num_patients:
                    break
            prev_loss = loss.item()
        with torch.no_grad():
            iter_start = perf_counter()
            individual_loss = self.compute_loss(us, hist_xs, hist_us, target_xs)
            loss = torch.sum(individual_loss)
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
        for (i, idx) in enumerate(idx_list):
            optimal_u.append(us_trajectory[idx][curr_action_idx:curr_action_idx + self.num_actions_list[i]])
            curr_action_idx += self.num_actions_list[i]
        optimal_u = torch.cat(optimal_u, dim=0).to(self.device)
        with torch.no_grad():
            self.us[:, -receding_horizon:] = optimal_u
        if self.is_logging:
            log = self.record_log(hist_xs=hist_xs.cpu().numpy(),
                                  hist_us=hist_us.cpu().numpy(),
                                  target_xs=target_xs.cpu().numpy(),
                                  individual_loss=individual_loss_trajectory,
                                  total_loss=total_loss_trajectory,
                                  time=np.array(time_trajectory),
                                  total_runtime=perf_counter() - loop_start,
                                  best_idx_list=idx_list,
                                  best_us=optimal_u.cpu().detach().numpy()
                                  )
        else:
            log = None
        return optimal_u, log

    def solve_fixed_initial(self, hist_xs, hist_us, target_xs):
        """
        :param hist_xs: [num_states x state_order x state dim]
        :param hist_us: [num_actions x action_order-1 x action dim]
        :param target_xs: [num_states x receding_horizon x state dim]
        :return:
        """
        loop_start = perf_counter()
        receding_horizon = target_xs.shape[1]
        with torch.no_grad():
            us = torch.empty(size=(self.num_actions, self.receding_horizon, self.action_dim)).to(self.device)
            nn.init.constant_(self.us, (self.u_max - self.u_min) / 2)
            us = torch.nn.Parameter(self.us[:, -receding_horizon:]).to(self.device)
        # opt = th_opt.Adam([us], **self.opt_config)
        opt = th_opt.Adagrad([us], **self.opt_config)
        # opt = th_opt.SGD([us], **self.opt_config)
        scheduler = th_opt.lr_scheduler.ReduceLROnPlateau(opt, **self.scheduler_config)
        individual_loss_trajectory = []
        total_loss_trajectory = []  # the loss over the optimization steps
        us_trajectory = []  # the optimized actions over the optimization steps
        time_trajectory = []  # the optimization runtime over the optimization steps
        for i in range(self.max_iter):
            iter_start = perf_counter()
            individual_loss = self.compute_loss(us, hist_xs, hist_us, target_xs)
            loss = torch.sum(individual_loss)
            individual_loss_trajectory.append(individual_loss.cpu().detach().numpy())
            total_loss_trajectory.append(loss.item())
            us_trajectory.append(us.data.cpu().detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step(loss)
            with torch.no_grad():
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            time_trajectory.append(perf_counter() - iter_start)
        with torch.no_grad():
            iter_start = perf_counter()
            individual_loss = self.compute_loss(us, hist_xs, hist_us, target_xs)
            loss = torch.sum(individual_loss)
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
        for (i, idx) in enumerate(idx_list):
            optimal_u.append(us_trajectory[idx][curr_action_idx:curr_action_idx + self.num_actions_list[i]])
            curr_action_idx += self.num_actions_list[i]
        optimal_u = torch.cat(optimal_u, dim=0).to(self.device)
        with torch.no_grad():
            self.us[:, -receding_horizon:] = optimal_u
        if self.is_logging:
            log = self.record_log(hist_xs=hist_xs.cpu().numpy(),
                                  hist_us=hist_us.cpu().numpy(),
                                  target_xs=target_xs.cpu().numpy(),
                                  individual_loss=individual_loss_trajectory,
                                  total_loss=total_loss_trajectory,
                                  time=np.array(time_trajectory),
                                  total_runtime=perf_counter() - loop_start,
                                  best_idx_list=idx_list,
                                  best_us=optimal_u.cpu().detach().numpy()
                                  )
        else:
            log = None
        return optimal_u, log

    def predict_future(self, hist_xs, hist_us, us):
        h0 = self.model.filter_history(self.graph, hist_xs, hist_us)
        prediction = self.model.multi_step_prediction(self.graph, h0, us)
        return prediction

    def predict_multiple_future(self, hist_xs, hist_us, us):
        h0 = self.model.filter_history(self.multiple_graph, hist_xs, hist_us)
        prediction = self.model.multi_step_prediction(self.multiple_graph, h0, us)
        return prediction
