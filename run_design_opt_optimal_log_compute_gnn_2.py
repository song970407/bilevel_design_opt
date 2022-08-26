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


def run_optimal_control(mpc_config, env_config, state_pos, action_pos):
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
    with open('data/control/design_opt/prb_config.pkl', 'rb') as f:
        prb_config = pickle.load(f)
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    with open('data/control/design_opt/target_config.pkl', 'rb') as f:
        target_config = pickle.load(f)

    num_x_list = [4]
    num_heaters_list = [5]
    num_repeats = 10
    r = 8
    start_idx = 58
    end_idx = 60
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
    model_names = ['GAT']
    data_ratio_list = [1.0]

    state_pos_list = []
    action_pos_list = []
    graph_list = []
    num_graphs = len(target_config['target_values_list'])

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            if not os.path.exists('design_opt_experiment/optimal/{}_{}/log_compute'.format(num_x, num_heaters)):
                os.makedirs('design_opt_experiment/optimal/{}_{}/log_compute'.format(num_x, num_heaters))
            for model_name in model_names:
                for data_ratio in data_ratio_list:
                    print('Now Num of x: {}, Num of Heater: {}, Repeat: {}, Model: {}_{}'.format(num_x, num_heaters,
                                                                                                 r, model_name,
                                                                                                 data_ratio))
                    with open('design_opt_experiment/{}_{}/design_opt_experiment_result_{}_{}_{}.pkl'.format(num_x,
                                                                                                             num_heaters,
                                                                                                             model_name,
                                                                                                             data_ratio,
                                                                                                             r),
                              'rb') as f:
                        design_opt_experiment_result = pickle.load(f)
                    graph = design_opt_experiment_result['optimized_graph']
                    state_pos = graph.nodes['state'].data['pos'].cpu().detach().numpy()
                    # action_pos = graph.nodes['action'].data['pos'].cpu().detach().numpy()
                    for itr in range(start_idx, end_idx):
                        print('Now Log Iteration: {}'.format(itr))
                        action_pos = design_opt_experiment_result['design_opt_log']['position_trajectory'][itr]
                        x_trajectory_list, u_trajectory_list, log_trajectory_list = run_optimal_control(mpc_config,
                                                                                                        env_config,
                                                                                                        state_pos,
                                                                                                        action_pos)
                        optimal_experiment_result = {
                            'x_trajectory_list': x_trajectory_list,
                            'u_trajectory_list': u_trajectory_list,
                            'log_trajectory_list': log_trajectory_list
                        }
                        with open(
                                'design_opt_experiment/optimal/{}_{}/log_compute/optimal_experiment_result_{}_{}_{}_{}.pkl'.format(
                                        num_x,
                                        num_heaters,
                                        model_name,
                                        data_ratio,
                                        r,
                                        itr), 'wb') as f:
                            pickle.dump(optimal_experiment_result, f)
