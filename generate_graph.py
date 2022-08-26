import os
import pickle

import numpy as np
import torch

from src.utils.get_graph import generate_full_graph
from src.utils.get_position import generate_random_position, generate_uniform_position
from src.utils.fix_seed import fix_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fix_seed()
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('data/control'):
    os.makedirs('data/control')
if not os.path.exists('data/control/mpc'):
    os.makedirs('data/control/mpc')
if not os.path.exists('data/control/design_opt'):
    os.makedirs('data/control/design_opt')


def generate_graph(data_dict):
    num_graphs = data_dict['num_graphs']
    num_sensors = data_dict['num_sensors']
    num_heaters = data_dict['num_heaters']
    domain_range = data_dict['domain_range']
    epsilon = data_dict['epsilon']
    state_pos_list = []
    action_pos_list = []
    graph_list = []
    for i in range(num_graphs):
        state_pos, action_pos = generate_random_position(domain_range, num_sensors, num_heaters, epsilon)
        torch_state_pos = torch.from_numpy(state_pos).float()
        torch_action_pos = torch.from_numpy(action_pos).float()
        graph = generate_full_graph(torch_state_pos, torch_action_pos)
        state_pos_list.append(state_pos)
        action_pos_list.append(action_pos)
        graph_list.append(graph)
    return state_pos_list, action_pos_list, graph_list


def generate_uniform_graph(graph_arg_config):
    num_x = graph_arg_config['num_x']
    num_heaters = graph_arg_config['num_heaters']
    domain_range = graph_arg_config['domain_range']
    epsilon = graph_arg_config['epsilon']

    state_pos, action_pos = generate_uniform_position(domain_range, num_x, num_x, num_heaters, epsilon)
    torch_state_pos = torch.from_numpy(state_pos).float()
    torch_action_pos = torch.from_numpy(action_pos).float()
    graph = generate_full_graph(torch_state_pos, torch_action_pos)
    return state_pos, action_pos, graph


def sample_target(num_targets, target_dist_config):
    target_length = target_dist_config['target_length']
    max_heatups = target_dist_config['max_heatups']
    target_range = target_dist_config['target_range']
    initial_target = target_dist_config['initial_target']
    num_heatups = np.random.randint(1, max_heatups + 1, size=num_targets)
    target_values_list = []
    target_times_list = []

    for idx in range(num_targets):
        each_length = int(target_length / num_heatups[idx])
        anneal_temp = np.sort(np.random.uniform(target_range[0], target_range[1], size=(num_heatups[idx])))
        heatup_ratio = np.random.randint(2, each_length - 2, size=(num_heatups[idx]))
        target_values = [initial_target]
        target_times = []
        for i in range(num_heatups[idx]):
            target_values.append(anneal_temp[i])
            target_values.append(anneal_temp[i])
            target_times.append(heatup_ratio[i])
            target_times.append(each_length - heatup_ratio[i])
        target_values_list.append(target_values)
        target_times_list.append(target_times)
    return target_values_list, target_times_list


if __name__ == '__main__':
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    domain_range = env_config['domain_range']
    epsilon = env_config['epsilon']
    """
    # Sample target Trajectory for Design Optimization and Test data
    """
    prb_config = {
        'ridge_coefficient': 0.0,
        'smoothness_coefficient': 0.0,
        'receding_history': 5,
        'receding_horizon': 10
    }
    with open('data/control/mpc/prb_config.pkl', 'wb') as f:
        pickle.dump(prb_config, f)
    with open('data/control/design_opt/prb_config.pkl', 'wb') as f:
        pickle.dump(prb_config, f)

    target_length = 24
    max_heatups = 4
    target_range = [0.0, 0.3]
    initial_target = 0.0

    target_dist_config = {
        'target_length': target_length,
        'max_heatups': max_heatups,
        'target_range': target_range,
        'initial_target': initial_target
    }
    with open('data/control/mpc/target_dist_config.pkl', 'wb') as f:
        pickle.dump(target_dist_config, f)
    with open('data/control/design_opt/target_dist_config.pkl', 'wb') as f:
        pickle.dump(target_dist_config, f)

    num_targets = 32
    target_values_list, target_times_list = sample_target(num_targets, target_dist_config)
    target_config = {
        'target_values_list': target_values_list,
        'target_times_list': target_times_list
    }
    with open('data/control/mpc/target_config.pkl', 'wb') as f:
        pickle.dump(target_config, f)
    with open('data/control/design_opt/target_config.pkl', 'wb') as f:
        pickle.dump(target_config, f)

    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]
    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            graph_arg_config = {
                'num_x': num_x,
                'num_heaters': num_heaters,
                'domain_range': domain_range,
                'epsilon': epsilon
            }
            state_pos_list = []
            action_pos_list = []
            graph_list = []
            for _ in range(num_targets):
                state_pos, action_pos, graph = generate_uniform_graph(graph_arg_config)
                state_pos_list.append(state_pos)
                action_pos_list.append(action_pos)
                graph_list.append(graph)
            graph_config = {
                'state_pos_list': state_pos_list,
                'action_pos_list': action_pos_list,
                'graph_list': graph_list
            }
            with open('data/control/mpc/graph_{}_{}.pkl'.format(num_x, num_heaters), 'wb') as f:
                pickle.dump(graph_config, f)

    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]
    num_repeats = 10

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            graph_arg_config = {
                'num_x': num_x,
                'num_heaters': num_heaters,
                'domain_range': domain_range,
                'epsilon': epsilon
            }
            for r in range(num_repeats):
                state_pos, action_pos, graph = generate_uniform_graph(graph_arg_config)
                graph_config = {
                    'state_pos': state_pos,
                    'action_pos': action_pos,
                    'graph': graph
                }
                with open('data/control/design_opt/graph_{}_{}_{}.pkl'.format(num_x, num_heaters, r), 'wb') as f:
                    pickle.dump(graph_config, f)
