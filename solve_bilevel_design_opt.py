import os
import argparse
import yaml
import pickle

import numpy as np
import torch
import dgl


from src.model.get_model import get_model
from src.solver.get_solver import get_solver
from src.utils.scale_data import minmax_scale
from src.utils.get_target import generate_target_trajectory
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def solve_bilevel_design_opt(args, env_config, bilevel_design_opt_problem_config, data_generation_config, data_preprocessing_config):
    solver_name = args.solver_name
    num_x = args.num_x
    num_heaters = args.num_heaters
    model_name = args.model_name

    domain_range = env_config['domain_range']
    epsilon = env_config['epsilon']
    data_saved_dir = bilevel_design_opt_problem_config['data_saved_dir']
    action_bound = data_generation_config['action_bound']
    scaler = data_preprocessing_config['scaler']
    state_scaler = data_preprocessing_config['state_scaler']
    action_scaler = data_preprocessing_config['action_scaler']

    model_config = yaml.safe_load(open('config/model/{}_config.yaml'.format(model_name), 'r'))
    m = get_model(model_name, model_config, True).to(device)
    m.eval()
    problem_data = pickle.load(open('{}/problem_{}_{}.pkl'.format(data_saved_dir, num_x, num_heaters), 'rb'))
    target_data = pickle.load(open('{}/target.pkl'.format(data_saved_dir), 'rb'))
    solver_config = yaml.safe_load(open('config/bilevel_design_opt_solver/{}_config.yaml'.format(solver_name), 'r'))
    solver_config['model'] = m
    solver_config['upper_bound'] = [domain_range[0] + epsilon, domain_range[1] + epsilon]
    solver_config['lower_bound'] = [minmax_scale(action_bound[0], action_scaler, scaler),
                                    minmax_scale(action_bound[1], action_scaler, scaler)]
    solver_config['device'] = device
    solver = get_solver(solver_name, solver_config)

    target = []
    for (target_value, target_time) in zip(target_data['target_values'], target_data['target_times']):
        target_ref = generate_target_trajectory(target_value, target_time)
        for _ in range(num_x**2):
            target.append(torch.tensor(target_ref, device=device))
    target = torch.stack(target, dim=0).unsqueeze(dim=-1)  # num_states_node x receding_horizon x 1
    opt_result = {
        'opt_action_pos': [],
        'opt_log': []
    }
    if not os.path.exists('opt_result/{}'.format(solver_name)):
        os.mkdir('opt_result/{}'.format(solver_name))
    for (state_pos, action_pos) in zip(problem_data['state_pos'], problem_data['action_pos']):
        # g = dgl.batch([graph.to(device) for _ in range(bilevel_design_opt_problem_config['num_targets'])])
        opt_action_pos, opt_log = solver.solve(target, state_pos, action_pos)
        opt_result['opt_action_pos'].append(opt_action_pos)
        opt_result['opt_log'].append(opt_log)
        pickle.dump(opt_result, open('opt_result/{}/{}_{}.pkl'.format(solver_name, num_x, num_heaters), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver_name', default='Implicit')
    parser.add_argument('--num_x', default=3)
    parser.add_argument('--num_heaters', default=4)
    parser.add_argument('--model_name', default='Linear')
    args = parser.parse_args()

    env_config = yaml.safe_load(open('config/env/env_config.yaml', 'r'))
    bilevel_design_opt_problem_config = yaml.safe_load(open('config/data/bilevel_design_opt_problem_config.yaml', 'r'))
    data_preprocessing_config = yaml.safe_load(open('config/data/data_generation_config.yaml', 'r'))
    solve_bilevel_design_opt(args, env_config, bilevel_design_opt_problem_config, data_preprocessing_config)
