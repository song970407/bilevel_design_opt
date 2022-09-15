import os
import argparse
import pickle

import numpy as np
import torch
import yaml

from src.env.HeatDiffusion import HeatDiffusionSystem
from src.solver.optimal_control import OptimalControl
from src.utils.get_target import generate_target_trajectory
from src.utils.fix_seed import fix_seed

fix_seed()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if not os.path.exists('bilevel_opt_result/optimal'):
    os.makedirs('bilevel_opt_result/optimal')


def run_optimal_control(mpc_config, env_config, data_generation_config, data_preprocessing_config, state_pos, action_pos):
    ridge_coefficient = mpc_config['ridge_coefficient']
    smoothness_coefficient = mpc_config['smoothness_coefficient']
    target_values_list = mpc_config['target_values_list']
    target_times_list = mpc_config['target_times_list']
    max_iter = mpc_config['max_iter']
    loss_threshold = mpc_config['loss_threshold']
    opt_config = mpc_config['opt_config']
    scheduler_config = mpc_config['scheduler_config']

    num_cells = env_config['num_cells']
    dt = env_config['dt']
    epsilon = env_config['epsilon']
    domain_range = env_config['domain_range']
    action_min = data_generation_config['action_bound'][0]
    action_max = data_generation_config['action_bound'][1]
    state_scaler = data_preprocessing_config['state_scaler']
    action_scaler = data_preprocessing_config['action_scaler']

    num_states = state_pos.shape[0]
    num_actions = action_pos.shape[0]

    u_min = action_min
    u_max = action_max
    is_logging = True

    target_list = []
    for (target_values, target_times) in zip(target_values_list, target_times_list):
        target = np.array(generate_target_trajectory(target_values, target_times))
        target = np.reshape(target, newshape=(-1, 1))
        target = np.concatenate([target for _ in range(num_states)], axis=1)
        target = torch.from_numpy(target).float().to(DEVICE)
        target_list.append(target)
    receding_horizon = target_list[0].shape[0]

    env = HeatDiffusionSystem(num_cells=num_cells,
                              dt=dt,
                              epsilon=epsilon,
                              domain_range=domain_range,
                              action_min=action_min,
                              action_max=action_max,
                              state_pos=state_pos,
                              action_pos=action_pos)
    env.reset()

    optimal_controller = OptimalControl(env, num_states, num_actions, receding_horizon, ridge_coefficient,
                                        smoothness_coefficient, u_min, u_max, max_iter, loss_threshold, is_logging,
                                        DEVICE, opt_config, scheduler_config)
    trajectory_x = []
    trajectory_u = []
    trajectory_log = []

    for (i, target) in enumerate(target_list):
        print('Now target number {}'.format(i))
        optimal_us, log = optimal_controller.solve(target)
        trajectory_log.append(log)
        x_traj = []
        x_traj.append(np.zeros((num_states)))
        u_traj = []
        env.reset()
        for optimal_u in optimal_us:
            x_traj.append(env.step(optimal_u))
            u_traj.append(optimal_u)
        x_traj = np.stack(x_traj)
        u_traj = np.stack(u_traj)
        trajectory_x.append(x_traj)
        trajectory_u.append(u_traj)
    return trajectory_x, trajectory_u, trajectory_log


if __name__ == '__main__':
    env_config = yaml.safe_load(open('config/env/env_config.yaml', 'r'))
    data_generation_config = yaml.safe_load(open('config/data/data_generation_config.yaml', 'r'))
    data_preprocessing_config = yaml.safe_load(open('config/data/data_preprocessing_config.yaml', 'r'))
    target = pickle.load(open('data/bilevel_design_opt/target.pkl', 'rb'))

    mpc_config = {
        'ridge_coefficient': 0,
        'smoothness_coefficient': 0,
        'target_values_list': target['target_values'],
        'target_times_list': target['target_times'],
        'max_iter': 200,
        'loss_threshold': 1e-9,
        'opt_config': {'lr': 2e-0},
        'scheduler_config': {'patience': 5, 'factor': 0.5, 'min_lr': 1e-4}
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_x', type=int, default=3)
    parser.add_argument('--num_heaters', type=int, default=5)
    parser.add_argument('--solver_name', type=str, default='cma_es')
    args = parser.parse_args()

    num_x = args.num_x
    num_heaters = args.num_heaters
    solver_name = args.solver_name

    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]

    if not os.path.exists('bilevel_opt_result/optimal/{}'.format(solver_name)):
        os.makedirs('bilevel_opt_result/optimal/{}'.format(solver_name))

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            opt_result = pickle.load(open('bilevel_opt_result/{}/{}_{}.pkl'.format(solver_name, num_x, num_heaters), 'rb'))
            state_pos = pickle.load(open('data/bilevel_design_opt/problem_{}_{}.pkl'.format(num_x, num_heaters), 'rb'))['state_pos'][0]
            action_pos = opt_result['opt_action_pos']
            x_trajectory_list, u_trajectory_list, log_trajectory_list = run_optimal_control(mpc_config,
                                                                                            env_config,
                                                                                            data_generation_config,
                                                                                            data_preprocessing_config,
                                                                                            state_pos,
                                                                                            action_pos)
            pickle.dump(mpc_config, open('bilevel_opt_result/optimal/{}/mpc_config_{}_{}.pkl'.format(solver_name, num_x, num_heaters), 'wb'))
            optimal_result = {
                'x_trajectory_list': x_trajectory_list,
                'u_trajectory_list': u_trajectory_list,
                'log_trajectory_list': log_trajectory_list
            }
            pickle.dump(optimal_result, open('bilevel_opt_result/optimal/{}/optimal_result_{}_{}.pkl'.format(solver_name, num_x, num_heaters), 'wb'))
