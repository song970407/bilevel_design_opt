import pickle

import numpy as np
import matplotlib.pyplot as plt

from src.utils.get_target import generate_target_trajectory


def measure_design_opt(result_list, model_names, num_x, num_heaters):
    fig, axes = plt.subplots(10, len(model_names), figsize=(3 * len(model_names), 30))
    axes_flatten = axes.flatten()
    for r in range(num_repeats):
        for i in range(len(model_names)):
            design_opt_experiment_result = result_list[model_names[i]][r]
            design_opt_log = design_opt_experiment_result['design_opt_log']
            total_loss_trajectory = design_opt_log['total_loss_trajectory']
            axes_flatten[len(model_names) * r + i].plot(design_opt_log['total_loss_trajectory'])
            axes_flatten[len(model_names) * r + i].set_yscale('log')
            axes_flatten[len(model_names) * r + i].set_title('{} Loss: {}'.format(model_names[i], np.round(np.min(total_loss_trajectory), 4)))
    fig.suptitle('{}, {}'.format(num_x ** 2, num_heaters))
    fig.tight_layout()
    fig.show()


def plot_graph(prev_graph_list, opt_graph_list, model_names):
    fig, axes = plt.subplots(10, 3, figsize=(9, 30))
    axes_flatten = axes.flatten()
    for r in range(num_repeats):
        axes_flatten[3 * r + 0].scatter(prev_graph_list[model_names[0]][r].nodes['state'].data['pos'][:, 0],
                                        prev_graph_list[model_names[0]][r].nodes['state'].data['pos'][:, 1], color='blue', label='Sensor')
        axes_flatten[3 * r + 0].scatter(prev_graph_list[model_names[0]][r].nodes['action'].data['pos'][:, 0],
                                        prev_graph_list[model_names[0]][r].nodes['action'].data['pos'][:, 1], color='red', label='Heater')
        axes_flatten[3 * r + 0].set_title('Before Optimizing')
        axes_flatten[3 * r + 0].set_xlim([-1, 1])
        axes_flatten[3 * r + 0].set_ylim([-1, 1])
        axes_flatten[3 * r + 0].legend()
        for (i, model_name) in enumerate(model_names):
            axes_flatten[3 * r + i + 1].scatter(opt_graph_list[model_name][r].nodes['state'].data['pos'][:, 0],
                                                opt_graph_list[model_name][r].nodes['state'].data['pos'][:, 1], color='blue', label='Sensor')
            axes_flatten[3 * r + i + 1].scatter(opt_graph_list[model_name][r].nodes['action'].data['pos'][:, 0],
                                                opt_graph_list[model_name][r].nodes['action'].data['pos'][:, 1], color='red', label='Heater')
            axes_flatten[3 * r + i + 1].set_title('Optimized by {}'.format(model_name))
            axes_flatten[3 * r + i + 1].set_xlim([-1, 1])
            axes_flatten[3 * r + i + 1].set_ylim([-1, 1])
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)

    model_names = ['GAT', 'ICGAT']
    data_ratio_list = [1.0]
    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]
    num_x_list = [3]
    num_heaters_list = [10, 20]
    num_repeats = 10

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            prev_graph_dict = {}
            opt_graph_dict = {}
            result_dict = {}
            for model_name in model_names:
                prev_graph_dict[model_name] = []
                opt_graph_dict[model_name] = []
                result_dict[model_name] = []

            for model_name in model_names:
                for data_ratio in data_ratio_list:
                    for r in range(num_repeats):
                        with open('data/control/design_opt/graph_{}_{}_{}.pkl'.format(num_x, num_heaters, r),
                                  'rb') as f:
                            prb_graph = pickle.load(f)
                        with open('design_opt_experiment/{}_{}/design_opt_experiment_result_{}_{}_{}.pkl'.format(num_x,
                                                                                                                 num_heaters,
                                                                                                                 model_name,
                                                                                                                 data_ratio,
                                                                                                                 r),
                                  'rb') as f:
                            design_opt_experiment_result = pickle.load(f)
                        result_dict[model_name].append(design_opt_experiment_result)
                        prev_graph_dict[model_name].append(design_opt_experiment_result['prev_graph'])
                        opt_graph_dict[model_name].append(design_opt_experiment_result['optimized_graph'])
            measure_design_opt(result_dict, model_names, num_x, num_heaters)
            plot_graph(prev_graph_dict, opt_graph_dict, model_names)
