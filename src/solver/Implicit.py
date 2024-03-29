from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as th_opt
import dgl

from src.utils.get_graph import generate_full_graph


def get_default_lower_opt_config():
    opt_config = {'lr': 1e-2}
    return opt_config


def get_default_lower_scheduler_config():
    scheduler_config = {'patience': 5, 'factor': 0.5, 'min_lr': 1e-4}
    return scheduler_config


def get_default_upper_opt_config():
    opt_config = {'lr': 1e-2}
    return opt_config


def get_default_upper_scheduler_config():
    scheduler_config = {'patience': 5, 'factor': 0.5, 'min_lr': 1e-4}
    return scheduler_config


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
                 is_logging: bool = True,
                 device: str = 'cpu',
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
        self.is_logging = is_logging
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
            # scheduler.step(loss)
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


class DesignOptimizer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 target_list: list,
                 num_states: int,
                 num_actions: int,
                 num_targets: int,
                 history_len: int,
                 ridge_coefficient: float,
                 smoothness_coefficient: float,
                 position_min: float,
                 position_max: float,
                 u_min: float,
                 u_max: float,
                 upper_max_iter: int,
                 lower_max_iter: int,
                 upper_loss_threshold: float,
                 lower_loss_threshold: float,
                 is_logging: bool = True,
                 upper_opt_config: dict = None,
                 upper_scheduler_config: dict = None,
                 lower_opt_config: dict = None,
                 lower_scheduler_config: dict = None,
                 device: str = 'cpu'):
        super(DesignOptimizer, self).__init__()
        self.lower_level_optimizer = LowerLevelOptimizer(model=model,
                                                         graph=graph,
                                                         num_states=num_states,
                                                         num_actions=num_actions,
                                                         history_len=history_len,
                                                         ridge_coefficient=ridge_coefficient,
                                                         smoothness_coefficient=smoothness_coefficient,
                                                         u_min=u_min,
                                                         u_max=u_max,
                                                         max_iter=lower_max_iter,
                                                         loss_threshold=lower_loss_threshold,
                                                         opt_config=lower_opt_config,
                                                         scheduler_config=lower_scheduler_config,
                                                         is_logging=is_logging,
                                                         device=device)

        self.model = model.to(device)
        self.graph = graph.to(device)
        self.target_list = target_list
        self.target = torch.cat(target_list, dim=0)
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_states_list = [num_states for _ in range(num_targets)]
        self.num_actions_list = [num_actions for _ in range(num_targets)]
        self.num_targets = num_targets
        self.total_states = num_states * num_targets
        self.total_actions = num_actions * num_targets
        self.history_len = history_len
        self.ridge_coefficient = ridge_coefficient
        self.smoothness_coefficient = smoothness_coefficient
        self.position_min = position_min
        self.position_max = position_max
        self.u_min = u_min
        self.u_max = u_max
        self.upper_max_iter = upper_max_iter
        self.lower_max_iter = lower_max_iter
        self.upper_loss_threshold = upper_loss_threshold
        self.lower_loss_threshold = lower_loss_threshold
        if upper_opt_config is None:
            upper_opt_config = get_default_lower_opt_config()
        self.upper_opt_config = upper_opt_config
        if upper_scheduler_config is None:
            upper_scheduler_config = get_default_lower_scheduler_config()
        self.upper_scheduler_config = upper_scheduler_config
        self.is_logging = is_logging
        self.device = device
        self.crit = torch.nn.MSELoss(reduction='none')
        self.criteria_smoothness = torch.nn.MSELoss(reduction='none')
        self.receding_horizon = self.target_list[0].shape[1]

        self.pos_dimension = self.graph.nodes['action'].data['pos'].shape[1]
        self.shape_position = (self.num_actions, self.pos_dimension)
        self.shape_us = (self.total_actions, self.receding_horizon, 1)
        self.num_position = self.num_actions * self.pos_dimension
        self.num_u = self.total_actions * self.receding_horizon * 1

    def set_position(self, position):
        with torch.no_grad():
            state_pos = self.graph.nodes['state'].data['pos'][:self.num_states]
        self.graph = dgl.batch([generate_full_graph(state_pos, position, self.device) for _ in range(self.num_targets)])

    def compute_individual_loss(self, position, us):
        self.hist_xs = torch.zeros((self.total_states, self.history_len, 1)).to(self.device)
        self.hist_us = torch.zeros((self.total_actions, self.history_len - 1, 1)).to(self.device)
        position = position.reshape(self.shape_position)
        us = us.reshape(self.shape_us)
        self.set_position(position)
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

    def compute_total_loss(self, position, us):
        return torch.mean(self.compute_individual_loss(position, us))

    def compute_hessian_batch(self, position, us):
        n = self.num_targets
        p = self.num_actions * self.pos_dimension
        q = self.num_actions * self.receding_horizon

        def jac_0(pos, u):
            jac = torch.autograd.functional.jacobian(func=self.compute_total_loss, inputs=(pos, u), create_graph=True)
            return jac[0]

        def jac_1(pos, u):
            jac = torch.autograd.functional.jacobian(func=self.compute_total_loss, inputs=(pos, u), create_graph=True)
            return jac[1].reshape(-1, q).sum(dim=0)

        (_hes_00, _hes_01) = torch.autograd.functional.jacobian(func=jac_0, inputs=(position, us))
        (_hes_10, _hes_11) = torch.autograd.functional.jacobian(func=jac_1, inputs=(position, us))

        hes_11 = torch.zeros(n * q, n * q, device=self.device)
        with torch.no_grad():
            hes_00 = _hes_00
            hes_01 = _hes_01
            hes_10 = _hes_01.transpose(0, 1)
            for i in range(n):
                hes_11[q * i: q * (i + 1), q * i: q * (i + 1)] = _hes_11[:, q * i: q * (i + 1)]
        return ((hes_00, hes_01), (hes_10, hes_11))

    def solve(self):
        loop_start = perf_counter()
        with torch.no_grad():
            initial_pos = self.graph.nodes['action'].data['pos'][:self.num_actions].reshape(-1)
        position = torch.nn.Parameter(initial_pos).to(self.device)
        position.data = position.data.clamp(min=self.position_min, max=self.position_max)
        opt = torch.optim.Adam([position], **self.upper_opt_config)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **self.upper_scheduler_config)
        total_loss_trajectory = []
        position_trajectory = []
        us_trajectory = []
        lower_level_log_trajectory = []
        for itr in range(self.upper_max_iter):
            start_time = perf_counter()
            self.set_position(position.reshape(self.shape_position))
            with torch.no_grad():
                position_trajectory.append(position.data.reshape(self.shape_position).cpu().detach().numpy())
            # Now Optimizing Start
            optimal_us, optimal_lambda_min, optimal_lambda_max, log = self.lower_level_optimizer.solve(
                position.reshape(self.shape_position), self.target_list)
            with torch.no_grad():
                total_loss_trajectory.append(log['best_loss'])
                lower_level_log_trajectory.append(log)
                us_trajectory.append(optimal_us.cpu().detach().numpy())
                optimal_us = optimal_us.reshape(-1)
                optimal_lambda_min = optimal_lambda_min.reshape(-1)
                optimal_lambda_max = optimal_lambda_max.reshape(-1)

                active_lambda_min = (self.u_min == optimal_us)  # same shape as optimal_u
                active_lambda_max = (self.u_max == optimal_us)  # same shape as optimal_u
                optimal_lambda = torch.cat(
                    [optimal_lambda_min[active_lambda_min], optimal_lambda_max[active_lambda_max]])
                num_of_active_constraints = (torch.sum(active_lambda_min) + torch.sum(active_lambda_max)).item()

            hessian_loss_raw = self.compute_hessian_batch(position, optimal_us)
            F_p = torch.cat(
                [hessian_loss_raw[1][0], torch.zeros((num_of_active_constraints, self.num_position)).to(self.device)],
                dim=0)
            G = torch.zeros((num_of_active_constraints, self.num_u)).to(self.device)
            count_idx = 0
            for k in range(optimal_lambda_min.shape[0]):
                if active_lambda_min[k] or active_lambda_max[k]:
                    G[count_idx, k] = 1
                    count_idx += 1
            F_x = torch.cat([torch.cat([hessian_loss_raw[1][1], G.transpose(0, 1)], dim=1), torch.cat(
                [G, torch.zeros((num_of_active_constraints, num_of_active_constraints)).to(self.device)], dim=1)],
                            dim=0)
            F_x_inverse = torch.linalg.inv(-F_x + 1e-6 * torch.eye(F_x.shape[0], device=self.device))
            u_p = torch.matmul(F_x_inverse, F_p)[:self.num_u]
            jacobian_loss = torch.autograd.functional.jacobian(self.compute_total_loss, (position, optimal_us))
            position.grad = torch.matmul(jacobian_loss[1], u_p) + jacobian_loss[0]
            opt.step()
            # scheduler.step(loss)
            with torch.no_grad():
                position.data = position.data.clamp(min=self.position_min, max=self.position_max)
            print('Iteration: {} Done, Time: {}'.format(itr, perf_counter() - start_time))

        self.set_position(position.reshape(self.shape_position))
        optimal_us, _, _, log = self.lower_level_optimizer.solve(
            position.reshape(self.shape_position), self.target_list)
        with torch.no_grad():
            total_loss_trajectory.append(log['best_loss'])
            position_trajectory.append(position.data.reshape(self.shape_position).cpu().detach().numpy())
            us_trajectory.append(optimal_us.cpu().detach().numpy())
            lower_level_log_trajectory.append(log)

        total_loss_trajectory = np.array(total_loss_trajectory)
        position_trajectory = np.array(position_trajectory)
        us_trajectory = np.array(us_trajectory)

        # Return the best position
        best_idx = np.argmin(total_loss_trajectory)
        best_loss = total_loss_trajectory[best_idx]
        best_position = position_trajectory[best_idx]
        best_us = us_trajectory[best_idx]
        log = {
            'total_loss_trajectory': total_loss_trajectory,
            'position_trajectory': position_trajectory,
            'us_trajectory': us_trajectory,
            'lower_level_log_trajectory': lower_level_log_trajectory,
            'best_idx': best_idx,
            'best_loss': best_loss,
            'best_position': best_position,
            'best_us': best_us
        }
        return log

    def predict_future(self, us):
        return self.model.multistep_prediction(self.graph, self.hist_xs, self.hist_us, us)


class ImplicitSolver(nn.Module):
    def __init__(self, solver_config):
        super(ImplicitSolver, self).__init__()
        self.history_len = solver_config['history_len']
        self.upper_max_iter = solver_config['upper_max_iter']
        self.lower_max_iter = solver_config['lower_max_iter']
        self.upper_opt_name = solver_config['upper_opt_name']
        self.upper_opt_config = solver_config['upper_opt_config']
        self.lower_opt_name = solver_config['lower_opt_name']
        self.lower_opt_config = solver_config['lower_opt_config']
        self.model = solver_config['model']
        self.upper_bound = solver_config['upper_bound']
        self.lower_bound = solver_config['lower_bound']
        self.device = solver_config['device']

        self.ridge_coefficient = 0.0
        self.smoothness_coefficient = 0.0
        self.position_min = solver_config['upper_bound'][0]
        self.position_max = solver_config['upper_bound'][1]
        self.u_min = solver_config['lower_bound'][0]
        self.u_max = solver_config['lower_bound'][1]

        self.crit = torch.nn.MSELoss(reduction='none')
        self.criteria_smoothness = torch.nn.MSELoss(reduction='none')

    def set_position(self, position):
        with torch.no_grad():
            state_pos = self.graph.nodes['state'].data['pos'][:self.num_states]
        self.graph = dgl.batch([generate_full_graph(state_pos, position, self.device) for _ in range(self.num_targets)])

    def compute_individual_loss(self, position, us):
        self.hist_xs = torch.zeros((self.total_states, self.history_len, 1)).to(self.device)
        self.hist_us = torch.zeros((self.total_actions, self.history_len - 1, 1)).to(self.device)
        position = position.reshape(self.shape_position)
        us = us.reshape(self.shape_us)
        self.set_position(position)
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

    def compute_total_loss(self, position, us):
        return torch.mean(self.compute_individual_loss(position, us))

    def compute_hessian_batch(self, position, us):
        n = self.num_targets
        p = self.num_actions * self.pos_dimension
        q = self.num_actions * self.receding_horizon

        def jac_0(pos, u):
            jac = torch.autograd.functional.jacobian(func=self.compute_total_loss, inputs=(pos, u), create_graph=True)
            return jac[0]

        def jac_1(pos, u):
            jac = torch.autograd.functional.jacobian(func=self.compute_total_loss, inputs=(pos, u), create_graph=True)
            return jac[1].reshape(-1, q).sum(dim=0)

        (_hes_00, _hes_01) = torch.autograd.functional.jacobian(func=jac_0, inputs=(position, us))
        (_hes_10, _hes_11) = torch.autograd.functional.jacobian(func=jac_1, inputs=(position, us))

        hes_11 = torch.zeros(n * q, n * q, device=self.device)
        with torch.no_grad():
            hes_00 = _hes_00
            hes_01 = _hes_01
            hes_10 = _hes_01.transpose(0, 1)
            for i in range(n):
                hes_11[q * i: q * (i + 1), q * i: q * (i + 1)] = _hes_11[:, q * i: q * (i + 1)]
        return ((hes_00, hes_01), (hes_10, hes_11))

    def _solve(self):
        loop_start = perf_counter()
        with torch.no_grad():
            initial_pos = self.graph.nodes['action'].data['pos'][:self.num_actions].reshape(-1)
        position = torch.nn.Parameter(initial_pos).to(self.device)
        position.data = position.data.clamp(min=self.position_min, max=self.position_max)
        opt = torch.optim.Adam([position], **self.upper_opt_config)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **self.upper_scheduler_config)
        total_loss_trajectory = []
        position_trajectory = []
        us_trajectory = []
        lower_level_log_trajectory = []
        for itr in range(self.upper_max_iter):
            start_time = perf_counter()
            self.set_position(position.reshape(self.shape_position))
            with torch.no_grad():
                position_trajectory.append(position.data.reshape(self.shape_position).cpu().detach().numpy())
            # Now Optimizing Start
            optimal_us, optimal_lambda_min, optimal_lambda_max, log = self.lower_level_optimizer.solve(
                position.reshape(self.shape_position), self.target_list)
            with torch.no_grad():
                total_loss_trajectory.append(log['best_loss'])
                lower_level_log_trajectory.append(log)
                us_trajectory.append(optimal_us.cpu().detach().numpy())
                optimal_us = optimal_us.reshape(-1)
                optimal_lambda_min = optimal_lambda_min.reshape(-1)
                optimal_lambda_max = optimal_lambda_max.reshape(-1)

                active_lambda_min = (self.u_min == optimal_us)  # same shape as optimal_u
                active_lambda_max = (self.u_max == optimal_us)  # same shape as optimal_u
                optimal_lambda = torch.cat(
                    [optimal_lambda_min[active_lambda_min], optimal_lambda_max[active_lambda_max]])
                num_of_active_constraints = (torch.sum(active_lambda_min) + torch.sum(active_lambda_max)).item()

            hessian_loss_raw = self.compute_hessian_batch(position, optimal_us)
            F_p = torch.cat(
                [hessian_loss_raw[1][0], torch.zeros((num_of_active_constraints, self.num_position)).to(self.device)],
                dim=0)
            G = torch.zeros((num_of_active_constraints, self.num_u)).to(self.device)
            count_idx = 0
            for k in range(optimal_lambda_min.shape[0]):
                if active_lambda_min[k] or active_lambda_max[k]:
                    G[count_idx, k] = 1
                    count_idx += 1
            F_x = torch.cat([torch.cat([hessian_loss_raw[1][1], G.transpose(0, 1)], dim=1), torch.cat(
                [G, torch.zeros((num_of_active_constraints, num_of_active_constraints)).to(self.device)], dim=1)],
                            dim=0)
            F_x_inverse = torch.linalg.inv(-F_x + 1e-6 * torch.eye(F_x.shape[0], device=self.device))
            u_p = torch.matmul(F_x_inverse, F_p)[:self.num_u]
            jacobian_loss = torch.autograd.functional.jacobian(self.compute_total_loss, (position, optimal_us))
            position.grad = torch.matmul(jacobian_loss[1], u_p) + jacobian_loss[0]
            opt.step()
            # scheduler.step(loss)
            with torch.no_grad():
                position.data = position.data.clamp(min=self.position_min, max=self.position_max)
            print('Iteration: {} Done, Time: {}'.format(itr, perf_counter() - start_time))

        self.set_position(position.reshape(self.shape_position))
        optimal_us, _, _, log = self.lower_level_optimizer.solve(
            position.reshape(self.shape_position), self.target_list)
        with torch.no_grad():
            total_loss_trajectory.append(log['best_loss'])
            position_trajectory.append(position.data.reshape(self.shape_position).cpu().detach().numpy())
            us_trajectory.append(optimal_us.cpu().detach().numpy())
            lower_level_log_trajectory.append(log)

        total_loss_trajectory = np.array(total_loss_trajectory)
        position_trajectory = np.array(position_trajectory)
        us_trajectory = np.array(us_trajectory)

        # Return the best position
        best_idx = np.argmin(total_loss_trajectory)
        best_loss = total_loss_trajectory[best_idx]
        best_position = position_trajectory[best_idx]
        best_us = us_trajectory[best_idx]
        log = {
            'total_loss_trajectory': total_loss_trajectory,
            'position_trajectory': position_trajectory,
            'us_trajectory': us_trajectory,
            'lower_level_log_trajectory': lower_level_log_trajectory,
            'best_idx': best_idx,
            'best_loss': best_loss,
            'best_position': best_position,
            'best_us': best_us
        }
        return best_position, log

    def solve_one(self, target_list, state_pos, action_pos):
        torch_state_pos = torch.from_numpy(state_pos).float().to(self.device)
        torch_action_pos = torch.from_numpy(action_pos).float().to(self.device)
        self.graph = generate_full_graph(torch_state_pos, torch_action_pos, self.device)
        self.target_list = target_list
        self.target = torch.cat(target_list, dim=0)
        self.num_states = state_pos.shape[0]
        self.num_actions = action_pos.shape[0]
        self.num_states_list = [self.num_states for _ in range(len(target_list))]
        self.num_actions_list = [self.num_actions for _ in range(len(target_list))]
        self.num_targets = len(target_list)
        self.total_states = self.num_states * self.num_targets
        self.total_actions = self.num_actions * self.num_targets
        self.receding_horizon = self.target_list[0].shape[1]

        self.pos_dimension = self.graph.nodes['action'].data['pos'].shape[1]
        self.shape_position = (self.num_actions, self.pos_dimension)
        self.shape_us = (self.total_actions, self.receding_horizon, 1)
        self.num_position = self.num_actions * self.pos_dimension
        self.num_u = self.total_actions * self.receding_horizon * 1

        self.lower_level_optimizer = LowerLevelOptimizer(model=self.model,
                                                         graph=self.graph,
                                                         num_states=self.num_states,
                                                         num_actions=self.num_actions,
                                                         history_len=self.history_len,
                                                         ridge_coefficient=self.ridge_coefficient,
                                                         smoothness_coefficient=self.smoothness_coefficient,
                                                         u_min=self.u_min,
                                                         u_max=self.u_max,
                                                         max_iter=self.lower_max_iter,
                                                         is_logging=True,
                                                         device=self.device,
                                                         opt_config=self.lower_opt_config, )

        return self._solve()

    def predict_future(self, us):
        return self.model.multistep_prediction(self.graph, self.hist_xs, self.hist_us, us)

    def solve(self, target_list, state_pos_list, action_pos_list):

        opt_action_pos_list = []
        opt_log_list = []
        for (state_pos, action_pos) in zip(state_pos_list, action_pos_list):
            opt_action_pos, opt_log = self.solve_one(target_list, state_pos, action_pos)
            opt_action_pos_list.append(opt_action_pos)
            opt_log_list.append(opt_log)

        opt_action_pos = None
        bilevel_opt_log = {
            'total_loss_trajectory': [],
            'position_trajectory': [],
            'us_trajectory': [],
            'lower_level_log_trajectory': [],
            'best_idx': None,
            'best_loss': float('inf'),
            'best_position': [],
            'best_us': []
        }
        for idx, opt_log in enumerate(opt_log_list):
            bilevel_opt_log['total_loss_trajectory'].append(opt_log['total_loss_trajectory'])
            bilevel_opt_log['position_trajectory'].append(opt_log['position_trajectory'])
            bilevel_opt_log['us_trajectory'].append(opt_log['us_trajectory'])
            bilevel_opt_log['lower_level_log_trajectory'].append(opt_log['lower_level_log_trajectory'])
            if opt_log['best_loss'] < bilevel_opt_log['best_loss']:
                bilevel_opt_log['best_loss'] = opt_log['best_loss']
                bilevel_opt_log['best_idx'] = np.array([idx, opt_log['best_idx']])
                opt_action_pos = opt_action_pos_list[idx]
            bilevel_opt_log['best_position'].append(opt_log['best_position'])
            bilevel_opt_log['best_us'].append(opt_log['best_us'])
        bilevel_opt_log['total_loss_trajectory'] = np.stack(bilevel_opt_log['total_loss_trajectory'], axis=1)
        bilevel_opt_log['position_trajectory'] = np.stack(bilevel_opt_log['position_trajectory'], axis=1)
        bilevel_opt_log['us_trajectory'] = np.stack(bilevel_opt_log['us_trajectory'], axis=1)
        bilevel_opt_log['best_position'] = np.stack(bilevel_opt_log['best_position'], axis=1)
        bilevel_opt_log['best_us'] = np.stack(bilevel_opt_log['best_us'], axis=1)
        return opt_action_pos, bilevel_opt_log
