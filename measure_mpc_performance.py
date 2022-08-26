import pickle

import numpy as np
import matplotlib.pyplot as plt

from src.utils.get_target import generate_target_trajectory
plt.style.use('seaborn')


def measure_mpc(mpc_config, experiment_result):
    ridge_coefficient = mpc_config['ridge_coefficient']
    smoothness_coefficient = mpc_config['smoothness_coefficient']
    target_values_list = mpc_config['target_values_list']
    target_times_list = mpc_config['target_times_list']

    x_trajectory_list = experiment_result['x_trajectory_list']
    u_trajectory_list = experiment_result['u_trajectory_list']

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
    model_names = ['GAT', 'ICGAT']
    data_ratio_list = [1.0]
    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]

    for num_x in num_x_list:
        loss_list_dict = {}
        loss_list_dict['optimal'] = []
        for model_name in model_names:
            loss_list_dict[model_name] = []
        for num_heaters in num_heaters_list:
            fig, axes = plt.subplots(1, 1)
            with open('mpc_experiment/optimal/{}_{}/mpc_config.pkl'.format(num_x, num_heaters), 'rb') as f:
                mpc_config = pickle.load(f)
            with open('mpc_experiment/optimal/{}_{}/optimal_experiment_result.pkl'.format(num_x, num_heaters), 'rb') as f:
                optimal_experiment_result = pickle.load(f)
            for model_name in model_names:
                for data_ratio in data_ratio_list:
                    with open('mpc_experiment/{}_{}/mpc_config_{}_{}.pkl'.format(num_x, num_heaters, model_name,
                                                                                 data_ratio), 'rb') as f:
                        mpc_config = pickle.load(f)
                    with open('mpc_experiment/{}_{}/mpc_experiment_result_{}_{}.pkl'.format(num_x,
                                                                                            num_heaters,
                                                                                            model_name,
                                                                                            data_ratio),
                              'rb') as f:
                        mpc_experiment_result = pickle.load(f)
                    opt_loss = measure_mpc(mpc_config, mpc_experiment_result)
                    loss_list_dict[model_name].append(np.mean(opt_loss))
                    print('Graph Size: ({}, {}), Model: {}, Opt. Loss Avg: {}, Std: {}'.format(num_x,
                                                                                               num_heaters,
                                                                                               model_name,
                                                                                               np.mean(opt_loss),
                                                                                               np.std(opt_loss),
                                                                                               ))
                    axes.plot(opt_loss, label='{}NN'.format(model_name[:-2]))
            opt_loss = measure_mpc(mpc_config, optimal_experiment_result)
            loss_list_dict['optimal'].append(np.mean(opt_loss))
            axes.plot(opt_loss, label='Optimal')
            print('Graph Size: ({}, {}), Optimal Control, Loss Avg: {}, Std: {}'.format(num_x,
                                                                                        num_heaters,
                                                                                        np.mean(opt_loss),
                                                                                        np.std(opt_loss),
                                                                                        ))
            axes.legend()
            axes.set_ylim([0.0, 0.3])
            fig.suptitle('Graph: ({}, {})'.format(num_x ** 2, num_heaters))
            fig.show()
        fig, axes = plt.subplots(1, 1)
        for model_name in model_names:
            axes.plot(num_heaters_list, loss_list_dict[model_name], label=model_name[:-2]+'NN')
        axes.plot(num_heaters_list, loss_list_dict['optimal'], label='Optimal')
        axes.legend()
        axes.set_xlabel('Number of Heaters')
        axes.set_ylabel('Average Control Loss')
        axes.set_title('Number of Sensors: {}'.format(num_x ** 2))
        fig.show()
