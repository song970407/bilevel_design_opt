from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as th_opt


def get_default_opt_config():
    opt_config = {'lr': 1e-2}
    return opt_config


def get_default_scheduler_config():
    scheduler_config = {'name': 'ReduceLROnPlateau', 'patience': 2, 'factor': 0.5, 'min_lr': 1e-3}
    return scheduler_config


class OptimalControl(nn.Module):
    def __init__(self,
                 env,
                 num_states: int,
                 num_actions: int,
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
        super(OptimalControl, self).__init__()

        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
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
            print("Initializing optimal control solver with {}".format(device))
        self.device = device
        # Initialize Control Variable
        self.crit = torch.nn.MSELoss(reduction='none')
        self.criteria_smoothness = torch.nn.MSELoss(reduction='none')
        self.learn_parameter()

    def learn_parameter(self):
        A_list = np.zeros((self.receding_horizon, self.num_states, self.num_actions))
        for i in range(self.num_actions):
            self.env.reset()
            a = np.zeros((self.num_actions))
            a[i] = 1.0
            A_list[0, :, i] = self.env.step(a)
            for t in range(self.receding_horizon - 1):
                A_list[t + 1, :, i] = self.env.step(np.zeros((self.num_actions)))
        self.A = torch.from_numpy(np.stack(A_list)).float().to(self.device)

    def predict_future(self, us):
        xs = []
        for t in range(self.receding_horizon):
            x = torch.zeros((self.num_states)).to(self.device)
            for k in range(t+1):
                x += torch.matmul(self.A[k], us[t-k])
            xs.append(x)
        return torch.stack(xs, dim=0)

    def compute_loss(self, us, target_xs):
        predicted_xs = self.predict_future(us)
        return self.crit(predicted_xs, target_xs).mean(dim=1).sum()

    @staticmethod
    def record_log(target_xs, loss, time, total_runtime, best_idx, best_us):
        log = {
            'target_xs': target_xs,
            'loss': loss,
            'time': time,
            'total_runtime': total_runtime,
            'best_idx': best_idx,
            'best_us': best_us
        }
        return log

    def solve(self, target_xs):
        loop_start = perf_counter()
        with torch.no_grad():
            us = torch.empty(size=(self.receding_horizon, self.num_actions)).to(self.device)
            nn.init.constant_(us, (self.u_max - self.u_min) / 2)
            us = torch.nn.Parameter(us).to(self.device)
        opt = th_opt.Adam([us], **self.opt_config)
        scheduler = th_opt.lr_scheduler.ReduceLROnPlateau(opt, **self.scheduler_config)
        loss_trajectory = []
        us_trajectory = []  # the optimized actions over the optimization steps
        time_trajectory = []  # the optimization runtime over the optimization steps

        for i in range(self.max_iter):
            iter_start = perf_counter()
            loss = self.compute_loss(us, target_xs)
            loss_trajectory.append(loss.cpu().detach().numpy())
            us_trajectory.append(us.data.cpu().detach().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step(loss)
            with torch.no_grad():
                us.data = us.data.clamp(min=self.u_min, max=self.u_max)
            time_trajectory.append(perf_counter() - iter_start)
        with torch.no_grad():
            iter_start = perf_counter()
            loss = self.compute_loss(us, target_xs)
            loss_trajectory.append(loss.cpu().detach().numpy())
            us_trajectory.append(us.data.cpu().detach().numpy())
            time_trajectory.append(perf_counter() - iter_start)
            # Return the best us
        loss_trajectory = np.stack(loss_trajectory, axis=0)
        best_idx = np.argmin(loss_trajectory)
        optimal_u = us_trajectory[best_idx]
        if self.is_logging:
            log = self.record_log(target_xs=target_xs.cpu().numpy(),
                                  loss=loss_trajectory,
                                  time=np.array(time_trajectory),
                                  total_runtime=perf_counter() - loop_start,
                                  best_idx=best_idx,
                                  best_us=optimal_u)
        else:
            log = None
        return optimal_u, log
