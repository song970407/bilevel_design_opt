import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils.get_target import generate_target_trajectory


def measure_design_opt_control(mpc_config, experiment_result):
    ridge_coefficient = mpc_config['ridge_coefficient']
    smoothness_coefficient = mpc_config['smoothness_coefficient']
    target_values_list = mpc_config['target_values_list']
    target_times_list = mpc_config['target_times_list']

    x_trajectory_list = experiment_result['x_trajectory_list']
    u_trajectory_list = experiment_result['u_trajectory_list']
    log_trajectory_list = experiment_result['log_trajectory_list']
    num_sensors = x_trajectory_list[0].shape[1]
    num_heaters = u_trajectory_list[0].shape[1]
    target_list = []
    for (target_values, target_times) in zip(target_values_list, target_times_list):
        target = np.array(generate_target_trajectory(target_values, target_times))
        target = np.reshape(target, newshape=(-1, 1))
        target = np.concatenate([target for _ in range(num_sensors)], axis=1)
        target_list.append(target)
    loss_list = []

    def compute_mpc_loss(x_traj, u_traj, target):
        loss_obj = np.square(x_traj[1:] - target).mean(axis=0).sum()
        loss_ridge = ridge_coefficient * np.square(u_traj).mean(axis=0).sum()
        u_prev = np.concatenate([np.zeros((1, num_heaters)), u_traj[:-1]], axis=0)
        loss_smoothness = smoothness_coefficient * np.square(u_traj - u_prev).mean(axis=0).sum()
        return loss_obj + loss_ridge + loss_smoothness

    """fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes_flatten = axes.flatten()
    for i in range(len(axes_flatten)):
        axes_flatten[i].plot(log_trajectory_list[i]['loss'])
        axes_flatten[i].set_yscale('log')
    fig.show()"""

    for (x_traj, u_traj, target) in zip(x_trajectory_list, u_trajectory_list, target_list):
        loss_list.append(compute_mpc_loss(x_traj, u_traj, target))
    return np.array(loss_list)


def plot_graph(optimized_graph, model_name):
    state_pos = optimized_graph.nodes['state'].data['pos'].cpu().detach().numpy()
    action_pos = optimized_graph.nodes['action'].data['pos'].cpu().detach().numpy()
    num_sensors = state_pos.shape[0]
    num_heaters = action_pos.shape[0]
    fig, axes = plt.subplots(1, 1)
    axes.scatter(state_pos[:, 0], state_pos[:, 1], label='Sensor', c='blue')
    axes.scatter(action_pos[:, 0], action_pos[:, 1], label='Heater', c='red')
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.legend()
    fig.suptitle('Graph: ({}, {}), Model: {}'.format(num_sensors, num_heaters, model_name))
    fig.show()


if __name__ == '__main__':
    with open('data/bilevel_design_opt/target.pkl', 'rb') as f:
        target = pickle.load(f)
    solver_names = ['cma_es', 'implicit_ICGAT', 'single_layer_ICGAT']
    # solver_names = ['cma_es']
    solver_names = ['implicit_Linear', 'implicit_GAT', 'implicit_ICGAT']
    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            fig, axes = plt.subplots(2, len(solver_names), figsize=(len(solver_names)*5, 10))
            axes_flatten = axes.flatten()
            for p_idx, solver_name in enumerate(solver_names):
                path = 'bilevel_opt_result/optimal/{}'.format(solver_name)
                mpc_config = pickle.load(open('{}/mpc_config_{}_{}.pkl'.format(path, num_x, num_heaters), 'rb'))
                bilevel_optimal_result = pickle.load(
                    open('{}/optimal_result_{}_{}.pkl'.format(path, num_x, num_heaters), 'rb'))
                opt_loss = measure_design_opt_control(mpc_config, bilevel_optimal_result)
                axes_flatten[p_idx].plot(opt_loss, label=solver_name)
                axes_flatten[p_idx].set_yscale('log')
                print('Graph Size: ({}, {}), Solver: {}, Loss: {}'.format(num_x,
                                                                          num_heaters,
                                                                          solver_name,
                                                                          100 * np.average(opt_loss)))
                bilevel_path = 'bilevel_opt_result/{}/{}_{}.pkl'.format(solver_name, num_x, num_heaters)
                bilevel_opt_result = pickle.load(open(bilevel_path, 'rb'))
                opt_action_pos = bilevel_opt_result['opt_action_pos']
                axes_flatten[len(solver_names) + p_idx].scatter(opt_action_pos[:, 0], opt_action_pos[:, 1])
                axes_flatten[len(solver_names) + p_idx].set_xlim([-1, 1])
                axes_flatten[len(solver_names) + p_idx].set_ylim([-1, 1])
            fig.suptitle('{}, {}'.format(num_x, num_heaters))
            fig.tight_layout()
            fig.show()
