import os
import yaml

import numpy as np
import torch

from src.env.HeatDiffusion import HeatDiffusionSystem
from src.utils.get_graph import generate_full_graph
from src.utils.get_position import generate_random_position
from src.utils.fix_seed import fix_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.exists('data'):
    os.makedirs('data')


def generate_data(data_config):
    num_traj = data_config['num_traj']
    traj_len = data_config['traj_len']
    num_cells = data_config['num_cells']
    dt = data_config['dt']
    epsilon = data_config['epsilon']
    domain_range = data_config['domain_range']
    action_min = data_config['action_min']
    action_max = data_config['action_max']
    num_sensors_bound = data_config['num_sensors_bound']
    num_actions_bound = data_config['num_actions_bound']

    state_trajectory_list = []
    action_trajectory_list = []
    graph_list = []
    for i in range(num_traj):
        print('Trajectory Number [{}] / [{}]'.format(i, num_traj))
        num_sensors = np.random.randint(num_sensors_bound[0], num_sensors_bound[1] + 1)
        num_heaters = np.random.randint(num_actions_bound[0], num_actions_bound[1] + 1)
        state_pos, action_pos = generate_random_position(domain_range, num_sensors, num_heaters, epsilon)
        torch_state_pos = torch.from_numpy(state_pos).float().to(device)
        torch_action_pos = torch.from_numpy(action_pos).float().to(device)
        env = HeatDiffusionSystem(num_cells=num_cells,
                                  dt=dt,
                                  epsilon=epsilon,
                                  domain_range=domain_range,
                                  action_min=action_min,
                                  action_max=action_max,
                                  state_pos=state_pos,
                                  action_pos=action_pos)
        state_trajectory, action_trajectory = env.generate_random_trajectory(traj_len)
        state_trajectory = torch.from_numpy(state_trajectory).float().cpu()  # Should be CPU version
        action_trajectory = torch.from_numpy(action_trajectory).float().cpu()  # Should be CPU version
        graph = generate_full_graph(torch_state_pos, torch_action_pos, device).cpu()
        state_trajectory_list.append(state_trajectory)
        action_trajectory_list.append(action_trajectory)
        graph_list.append(graph)
    return state_trajectory_list, action_trajectory_list, graph_list


def generate_heat_diffusion_data(env_config, data_generation_config):
    num_cells = env_config['num_cells']
    dt = env_config['dt']
    epsilon = env_config['epsilon']
    domain_range = env_config['domain_range']

    traj_len = data_generation_config['traj_len']
    num_train_traj = data_generation_config['num_train_traj']
    num_val_traj = data_generation_config['num_val_traj']
    num_test_traj = data_generation_config['num_test_traj']
    num_sensors_bound = data_generation_config['num_sensors_bound']
    num_heaters_bound = data_generation_config['num_heaters_bound']
    action_min = data_generation_config['action_min']
    action_max = data_generation_config['action_max']
    train_data_saved_path = data_generation_config['train_data_saved_path']
    val_data_saved_path = data_generation_config['val_data_saved_path']
    test_data_saved_path = data_generation_config['test_data_saved_path']
    seed_num = data_generation_config['seed_num']
    fix_seed(seed_num)

    def _generate_data(num_traj, saved_path):
        data = []
        for _ in range(num_traj):
            episode = {}
            num_sensors = np.random.randint(num_sensors_bound[0], num_sensors_bound[1] + 1)
            num_heaters = np.random.randint(num_heaters_bound[0], num_heaters_bound[1] + 1)
            state_pos, action_pos = generate_random_position(domain_range, num_sensors, num_heaters, epsilon)
            env = HeatDiffusionSystem(num_cells=num_cells,
                                      dt=dt,
                                      epsilon=epsilon,
                                      domain_range=domain_range,
                                      action_min=action_min,
                                      action_max=action_max,
                                      state_pos=state_pos,
                                      action_pos=action_pos)
            state_trajectory, action_trajectory = env.generate_random_trajectory(traj_len)
            episode = {
                'state_pos': state_pos,
                'action_pos': action_pos,
                'state_trajectory': state_trajectory,
                'action_trajectory': action_trajectory
            }
            data.append(episode)
        with open(saved_path, 'w') as _f:
            yaml.dump(data, _f)

    _generate_data(num_train_traj, train_data_saved_path)
    _generate_data(num_val_traj, val_data_saved_path)
    _generate_data(num_test_traj, test_data_saved_path)


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


if __name__ == '__main__':
    with open('config/env/env_config.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    with open('config/data/data_generation_config.yaml', 'r') as f:
        data_generation_config = yaml.safe_load(f)
    generate_heat_diffusion_data(env_config, data_generation_config)
