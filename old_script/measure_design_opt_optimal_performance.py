import pickle

import numpy as np
import matplotlib.pyplot as plt

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
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    with open('data/control/design_opt/target_config.pkl', 'rb') as f:
        target_config = pickle.load(f)

    model_names = ['GAT', 'ICGAT']
    data_ratio_list = [1.0]
    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]
    # num_x_list = [3]
    # num_heaters_list = [15, 20]
    num_repeats = 10

    """for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            loss_dict = {}
            loss_list_dict = {}
            for model_name in model_names:
                for data_ratio in data_ratio_list:
                    loss_dict[model_name] = []
                    print('Now {}'.format(model_name))
                    for r in range(num_repeats):
                        with open('design_opt_experiment/optimal/{}_{}/mpc_config_{}_{}_{}.pkl'.format(num_x,
                                                                                                       num_heaters,
                                                                                                       model_name,
                                                                                                       data_ratio,
                                                                                                       r),
                                  'rb') as f:
                            mpc_config = pickle.load(f)
                        with open('design_opt_experiment/optimal/{}_{}/optimal_experiment_result_{}_{}_{}.pkl'.format(
                                num_x,
                                num_heaters,
                                model_name,
                                data_ratio, r),
                                'rb') as f:
                            optimal_experiment_result = pickle.load(f)
                        opt_loss = measure_design_opt_control(mpc_config, optimal_experiment_result)
                        loss_dict[model_name].append(np.mean(opt_loss))
                        loss_list_dict['{}_{}'.format(model_name, r)] = opt_loss
                        # loss_list_dict['{}_Optimal'.format(model_name)].append(np.mean(opt_loss))
                    print('Graph Size: ({}, {}), Model: {}, Control Loss Best: {}'.format(num_x,
                                                                                          num_heaters,
                                                                                          model_name,
                                                                                          np.min(
                                                                                              loss_dict[model_name])))
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            axes_flatten = axes.flatten()
            for r in range(num_repeats):
                for model_name in model_names:
                    axes_flatten[r].plot(loss_list_dict['{}_{}'.format(model_name, r)], label=model_name)
                    axes_flatten[r].set_ylim([0.0, 0.02])
            axes_flatten[-1].legend()
            fig.show()"""

    for num_x in num_x_list:
        fig1, axes1 = plt.subplots(1, 1)
        fig2, axes2 = plt.subplots(1, 1)
        for num_heaters in num_heaters_list:
            loss_dict = {}
            loss_list_dict = {}
            for model_name in model_names:
                for data_ratio in data_ratio_list:
                    loss_list = []
                    for r in range(num_repeats):
                        with open('design_opt_experiment/{}_{}/design_opt_experiment_result_{}_{}_{}.pkl'.format(num_x,
                                                                                                                 num_heaters,
                                                                                                                 model_name,
                                                                                                                 data_ratio,
                                                                                                                 r),
                                  'rb') as f:
                            design_opt_experiment_result = pickle.load(f)
                        loss_list.append(
                            np.min(design_opt_experiment_result['design_opt_log']['total_loss_trajectory']))
                    r = np.argmin(loss_list)

                    with open('design_opt_experiment/optimal/{}_{}/mpc_config_{}_{}_{}.pkl'.format(num_x,
                                                                                                   num_heaters,
                                                                                                   model_name,
                                                                                                   data_ratio,
                                                                                                   r),
                              'rb') as f:
                        mpc_config = pickle.load(f)
                    with open('design_opt_experiment/optimal/{}_{}/optimal_experiment_result_{}_{}_{}.pkl'.format(
                            num_x,
                            num_heaters,
                            model_name,
                            data_ratio, r),
                            'rb') as f:
                        optimal_experiment_result = pickle.load(f)
                    opt_loss = measure_design_opt_control(mpc_config, optimal_experiment_result)
                    if model_name == 'GAT':
                        axes1.plot(opt_loss, label=num_heaters)
                    else:
                        axes2.plot(opt_loss, label=num_heaters)
                    print('Graph Size: ({}, {}), Model: {}, Best Idx: {}, Loss: {}'.format(num_x,
                                                                                           num_heaters,
                                                                                           model_name,
                                                                                           r,
                                                                                           np.average(opt_loss)))

                    with open('design_opt_experiment/{}_{}/design_opt_experiment_result_{}_{}_{}.pkl'.format(num_x,
                                                                                                             num_heaters,
                                                                                                             model_name,
                                                                                                             data_ratio,
                                                                                                             r),
                              'rb') as f:
                        design_opt_experiment_result = pickle.load(f)
        axes1.legend()
        axes1.set_yscale('log')
        fig1.suptitle('{}, GNN'.format(num_x))
        fig1.tight_layout()
        fig1.show()
        axes2.legend()
        axes2.set_yscale('log')
        fig2.suptitle('{}, ICGNN'.format(num_x))
        fig2.tight_layout()
        fig2.show()
