import os
import pickle

import numpy as np
import torch

from src.env.HeatDiffusion import HeatDiffusionSystem
from src.control.optimal_control import OptimalControl
from src.utils.get_target import generate_target_trajectory
from src.utils.fix_seed import fix_seed

fix_seed()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if not os.path.exists('design_opt_experiment/optimal'):
    os.makedirs('design_opt_experiment/optimal')


def run_optimal_control(mpc_config, env_config, state_pos_list, action_pos_list):
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
    action_min = env_config['action_min']
    action_max = env_config['action_max']
    state_scaler = env_config['state_scaler']
    action_scaler = env_config['action_scaler']

    num_states = state_pos_list[0].shape[0]
    num_actions = action_pos_list[0].shape[0]

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

    env_list = []
    for (state_pos, action_pos) in zip(state_pos_list, action_pos_list):
        env = HeatDiffusionSystem(num_cells=num_cells,
                                  dt=dt,
                                  epsilon=epsilon,
                                  domain_range=domain_range,
                                  action_min=action_min,
                                  action_max=action_max,
                                  state_pos=state_pos,
                                  action_pos=action_pos)
        env.reset()
        env_list.append(env)

    trajectory_x = []
    trajectory_u = []
    trajectory_log = []

    for (i, (env, target)) in enumerate(zip(env_list, target_list)):
        print('Now Number {}'.format(i))
        optimal_controller = OptimalControl(env, num_states, num_actions, receding_horizon, ridge_coefficient,
                                            smoothness_coefficient, u_min, u_max, max_iter, loss_threshold, is_logging,
                                            DEVICE, opt_config, scheduler_config)
        optimal_us, log = optimal_controller.solve(target)
        trajectory_log.append(log)
        x_traj = []
        u_traj = []
        x_traj.append(np.zeros((num_states)))
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
    with open('data/control/mpc/prb_config.pkl', 'rb') as f:
        prb_config = pickle.load(f)
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    with open('data/control/mpc/target_config.pkl', 'rb') as f:
        target_config = pickle.load(f)

    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]
    mpc_config = {
        'ridge_coefficient': prb_config['ridge_coefficient'],
        'smoothness_coefficient': prb_config['smoothness_coefficient'],
        'target_values_list': target_config['target_values_list'],
        'target_times_list': target_config['target_times_list'],
        'max_iter': 200,
        'loss_threshold': 1e-9,
        'opt_config': {'lr': 2e-0},
        'scheduler_config': {'patience': 5, 'factor': 0.5, 'min_lr': 1e-4}
    }

    graph_list = []
    num_graphs = len(target_config['target_values_list'])

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            if not os.path.exists('mpc_experiment/optimal/{}_{}'.format(num_x, num_heaters)):
                os.makedirs('mpc_experiment/optimal/{}_{}'.format(num_x, num_heaters))
            print('Now Num of x: {}, Num of Heater: {}'.format(num_x, num_heaters))
            with open('data/control/mpc/graph_{}_{}.pkl'.format(num_x, num_heaters), 'rb') as f:
                graph_config = pickle.load(f)
            state_pos_list = graph_config['state_pos_list']
            action_pos_list = graph_config['action_pos_list']
            with open('mpc_experiment/optimal/mpc_config.pkl'.format(num_x, num_heaters), 'wb') as f:
                pickle.dump(mpc_config, f)
            x_trajectory_list, u_trajectory_list, log_trajectory_list = run_optimal_control(mpc_config,
                                                                                            env_config,
                                                                                            state_pos_list,
                                                                                            action_pos_list)
            with open('mpc_experiment/optimal/{}_{}/mpc_config.pkl'.format(num_x, num_heaters), 'wb') as f:
                pickle.dump(mpc_config, f)

            optimal_experiment_result = {
                'x_trajectory_list': x_trajectory_list,
                'u_trajectory_list': u_trajectory_list,
                'log_trajectory_list': log_trajectory_list
            }
            with open('mpc_experiment/optimal/{}_{}/optimal_experiment_result.pkl'.format(num_x, num_heaters),
                      'wb') as f:
                pickle.dump(optimal_experiment_result, f)
