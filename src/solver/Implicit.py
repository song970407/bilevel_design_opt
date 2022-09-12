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
                 receding_history: int,
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
        self.receding_history = receding_history
        self.ridge_coefficient = ridge_coefficient
        self.smoothness_coefficient = smoothness_coefficient
        self.u_min = u_min
        self.u_max = u_max
        self.max_iter = max_iter
        self.loss_threshold = loss_threshold
        self.is_logging = is_logging
        self.opt_config = opt_config
        self.scheduler_config = scheduler_config

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
        self.hist_xs = torch.zeros((self.total_states, self.receding_history, 1)).to(self.device)
        self.hist_us = torch.zeros((self.total_actions, self.receding_history - 1, 1)).to(self.device)
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
        scheduler = th_opt.lr_scheduler.ReduceLROnPlateau(opt, **self.scheduler_config)

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
        h0 = self.model.filter_history(self.graph, self.hist_xs, self.hist_us)
        prediction = self.model.multi_step_prediction(self.graph, h0, us)
        return prediction


class DesignOptimizer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 target_list: list,
                 num_states: int,
                 num_actions: int,
                 num_targets: int,
                 receding_history: int,
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
                                                         receding_history=receding_history,
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
        self.receding_history = receding_history
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
        self.hist_xs = torch.zeros((self.total_states, self.receding_history, 1)).to(self.device)
        self.hist_us = torch.zeros((self.total_actions, self.receding_history - 1, 1)).to(self.device)
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
        h0 = self.model.filter_history(self.graph, self.hist_xs, self.hist_us)
        prediction = self.model.multi_step_prediction(self.graph, h0, us)
        return prediction


# Should be Fixed
class LowerLevelOptimizerParallel(LowerLevelOptimizer):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 target: torch.tensor,
                 num_states: int,
                 num_actions: int,
                 num_targets: int,
                 num_graphs: int,
                 receding_history: int,
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
        super(LowerLevelOptimizerParallel, self).__init__(model=model,
                                                          graph=graph,
                                                          target=target,
                                                          num_states=num_states,
                                                          num_actions=num_actions,
                                                          num_targets=num_targets,
                                                          receding_history=receding_history,
                                                          ridge_coefficient=ridge_coefficient,
                                                          smoothness_coefficient=smoothness_coefficient,
                                                          u_min=u_min,
                                                          u_max=u_max,
                                                          max_iter=max_iter,
                                                          loss_threshold=loss_threshold,
                                                          opt_config=opt_config,
                                                          scheduler_config=scheduler_config,
                                                          is_logging=is_logging,
                                                          device=device)
        self.num_graphs = num_graphs

    def set_position(self, position):
        with torch.no_grad():
            state_pos = self.graph.nodes['state'].data['pos']
            graph_list = []
            for i in range(self.num_graphs):
                graph_list.append(generate_full_graph(state_pos[int(self.num_states * i / self.num_graphs): int(self.num_states * (i + 1) / self.num_graphs)],
                                                      position[int(self.num_actions * i / self.num_graphs): int(self.num_actions * (i + 1) / self.num_graphs)],
                                                      self.device))
            graph = dgl.batch(graph_list)
            self.graph = dgl.batch([graph for _ in range(self.num_targets)])


class DesignOptimizerParallel(DesignOptimizer):
    def __init__(self,
                 model: nn.Module,
                 graph: dgl.graph,
                 target: torch.tensor,
                 num_states: int,
                 num_actions: int,
                 num_targets: int,
                 num_graphs: int,
                 receding_history: int,
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
        super(DesignOptimizerParallel, self).__init__(model=model,
                                                      graph=graph,
                                                      target=target,
                                                      num_states=num_states,
                                                      num_actions=num_actions,
                                                      num_targets=num_targets,
                                                      receding_history=receding_history,
                                                      ridge_coefficient=ridge_coefficient,
                                                      smoothness_coefficient=smoothness_coefficient,
                                                      position_min=position_min,
                                                      position_max=position_max,
                                                      u_min=u_min,
                                                      u_max=u_max,
                                                      upper_max_iter=upper_max_iter,
                                                      lower_max_iter=lower_max_iter,
                                                      upper_loss_threshold=upper_loss_threshold,
                                                      lower_loss_threshold=lower_loss_threshold,
                                                      upper_opt_config=upper_opt_config,
                                                      upper_scheduler_config=upper_scheduler_config,
                                                      lower_opt_config=lower_opt_config,
                                                      lower_scheduler_config=lower_scheduler_config,
                                                      is_logging=is_logging,
                                                      device=device)
        self.lower_level_optimizer = LowerLevelOptimizerParallel(model=model,
                                                                 graph=graph,
                                                                 target=target,
                                                                 num_states=num_states,
                                                                 num_actions=num_actions,
                                                                 num_targets=num_targets,
                                                                 num_graphs=num_graphs,
                                                                 receding_history=receding_history,
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
        self.num_graphs = num_graphs

    def set_position(self, position):
        with torch.no_grad():
            state_pos = self.graph.nodes['state'].data['pos']
        graph_list = []
        for i in range(self.num_graphs):
            graph_list.append(generate_full_graph(state_pos[int(self.num_states * i / self.num_graphs): int(self.num_states * (i + 1) / self.num_graphs)],
                                                  position[int(self.num_actions * i / self.num_graphs): int(self.num_actions * (i + 1) / self.num_graphs)],
                                                  self.device))
        graph = dgl.batch(graph_list)
        self.graph = dgl.batch([graph for _ in range(self.num_targets)])



class ImplicitSolver(nn.Module):
    def __init__(self, solver_config):
        super(ImplicitSolver, self).__init__()