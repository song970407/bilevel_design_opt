import yaml
import pickle
from time import perf_counter

import numpy as np
import torch

from src.env.HeatDiffusion import HeatDiffusionSystem
from src.utils.get_graph import generate_full_graph
from src.utils.get_position import generate_random_position, generate_uniform_position
from src.utils.fix_seed import fix_seed


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
        for i in range(num_traj):
            start_time = perf_counter()
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
            torch_state_pos = torch.from_numpy(state_pos).float()
            torch_action_pos = torch.from_numpy(action_pos).float()
            graph = generate_full_graph(torch_state_pos, torch_action_pos)

            episode = {
                'state_pos': state_pos,
                'action_pos': action_pos,
                'state_trajectory': state_trajectory,
                'action_trajectory': action_trajectory,
                'graph': graph
            }
            data.append(episode)
            print('[{}] / [{}] Done, Time: {}'.format(i+1, num_traj, perf_counter() - start_time))
        pickle.dump(data, open(saved_path, 'wb'))

    _generate_data(num_train_traj, train_data_saved_path)
    _generate_data(num_val_traj, val_data_saved_path)
    _generate_data(num_test_traj, test_data_saved_path)


def generate_bilevel_design_opt_problem(bilevel_design_opt_problem_config):
    num_x_list = bilevel_design_opt_problem_config['num_x_list']
    num_heaters_list = bilevel_design_opt_problem_config['num_heaters_list']
    num_graphs = bilevel_design_opt_problem_config['num_repeats']
    domain_range = bilevel_design_opt_problem_config['domain_range']
    epsilon = bilevel_design_opt_problem_config['epsilon']
    num_targets = bilevel_design_opt_problem_config['num_targets']
    target_length = bilevel_design_opt_problem_config['target_length']
    max_heatups = bilevel_design_opt_problem_config['max_heatups']
    target_range = bilevel_design_opt_problem_config['target_range']
    initial_target = bilevel_design_opt_problem_config['initial_target']
    saved_dir = bilevel_design_opt_problem_config['saved_dir']
    seed_num = bilevel_design_opt_problem_config['seed_num']
    fix_seed(seed_num)

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            problem = {
                'state_pos': [],
                'action_pos': [],
                'graph': []
            }
            state_pos_list = []
            action_pos_list = []
            graph_list = []
            for graph_idx in range(num_graphs):
                state_pos, action_pos = generate_uniform_position(domain_range, num_x, num_x, num_heaters, epsilon)
                torch_state_pos = torch.from_numpy(state_pos).float()
                torch_action_pos = torch.from_numpy(action_pos).float()
                graph = generate_full_graph(torch_state_pos, torch_action_pos)
                problem['state_pos'].append(state_pos)
                problem['action_pos'].append(action_pos)
                problem['graph'].append(graph)
            pickle.dump(problem, open('{}/problem_{}_{}.pkl'.format(saved_dir, num_x, num_heaters), 'wb'))

    target_values_list = []
    target_times_list = []
    target = {
        'target_values': [],
        'target_times': []
    }
    for idx in range(num_targets):
        num_heatups = np.random.randint(1, max_heatups + 1, size=num_targets)
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
        target['target_values'].append(target_values)
        target['target_times'].append(target_times)
    pickle.dump(target, open('{}/target.pkl'.format(saved_dir), 'wb'))


if __name__ == '__main__':
    env_config = yaml.safe_load(open('config/env/env_config.yaml', 'r'))
    data_generation_config = yaml.safe_load(open('config/data/data_generation_config.yaml', 'r'))
    bilevel_design_opt_problem_config = yaml.safe_load(open('config/data/bilevel_design_opt_problem_config.yaml', 'r'))
    generate_heat_diffusion_data(env_config, data_generation_config)
    generate_bilevel_design_opt_problem(bilevel_design_opt_problem_config)
