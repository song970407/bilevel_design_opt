import os
import pickle
from time import perf_counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import dgl

from src.env.HeatDiffusion import HeatDiffusionSystem
from src.model.get_model import get_model
from src.control.mpc import MPC
from src.utils.get_target import generate_target_trajectory
from src.utils.fix_seed import fix_seed

fix_seed()
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if not os.path.exists('design_opt_experiment/test'):
    os.makedirs('design_opt_experiment/test')


def run_mpc(mpc_config, env_config, model_config, graph):
    ridge_coefficient = mpc_config['ridge_coefficient']
    smoothness_coefficient = mpc_config['smoothness_coefficient']
    target_values_list = mpc_config['target_values_list']
    target_times_list = mpc_config['target_times_list']
    receding_horizon = mpc_config['receding_horizon']
    receding_history = mpc_config['receding_history']
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

    # Set Model Information
    model_name = model_config['model_name']
    hidden_dim = model_config['hidden_dim']
    is_convex = model_config['is_convex']
    data_ratio = model_config['data_ratio']
    saved_model_path = 'saved_model/{}_{}.pt'.format(model_name, data_ratio)
    model = get_model(model_name, hidden_dim, is_convex, saved_model_path).to(DEVICE)

    state_pos = graph.nodes['state'].data['pos']
    action_pos = graph.nodes['action'].data['pos']
    num_sensors = state_pos.shape[0]
    num_heaters = action_pos.shape[0]

    state_dim = 1
    action_dim = 1
    u_min = (action_min - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
    u_max = (action_max - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
    is_logging = True

    num_targets = len(target_values_list)
    target_list = []
    for (target_values, target_times) in zip(target_values_list, target_times_list):
        target = np.array(generate_target_trajectory(target_values, target_times))
        target = np.reshape(target, newshape=(1, -1, 1))
        target = np.concatenate([target for _ in range(num_sensors)], axis=0)
        target_list.append(target)
    target = np.concatenate(target_list, axis=0)
    target = torch.from_numpy(target).float().to(DEVICE)

    env_list = []
    num_states = 0
    num_actions = 0
    num_states_list = []
    num_actions_list = []
    state_split_list = []
    action_split_list = []

    for i in range(num_targets):
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
        num_states += state_pos.shape[0]
        num_actions += action_pos.shape[0]
        num_states_list.append(state_pos.shape[0])
        num_actions_list.append(action_pos.shape[0])
        if i < num_targets - 1:
            state_split_list.append(num_states)
            action_split_list.append(num_actions)

    graph = dgl.batch([graph for _ in range(num_targets)])
    mpc_controller = MPC(model, graph, num_states, num_actions, num_states_list, num_actions_list, state_dim,
                         action_dim, ridge_coefficient, smoothness_coefficient, u_min, u_max,
                         max_iter, loss_threshold, is_logging, DEVICE, opt_config, scheduler_config)
    hist_xs = torch.zeros((num_states, receding_history, state_dim)).to(DEVICE)
    hist_us = torch.zeros((num_actions, receding_history - 1, action_dim)).to(DEVICE)
    trajectory_x = []
    trajectory_x.append(np.zeros(num_states))
    trajectory_u = []
    trajectory_log = []

    with graph.local_scope():
        for t in range(target.shape[1]):
            start_time = perf_counter()
            action, log = mpc_controller.solve(hist_xs, hist_us, target[:, t:t + receding_horizon])
            print('Timestep [{}] / [{}], MPC Optimization Done, Computation Time: {}'.format(t + 1, target.shape[1],
                                                                                             perf_counter() - start_time))
            curr_u = action[:, 0, 0].cpu().detach().numpy()
            curr_u = curr_u * (action_scaler[1] - action_scaler[0]) + action_scaler[0]
            curr_u_list = np.split(curr_u, action_split_list)
            curr_x = []
            for i in range(num_targets):
                curr_x.append(env_list[i].step(curr_u_list[i]))
            curr_x = np.concatenate(curr_x)
            trajectory_x.append(curr_x)
            trajectory_u.append(curr_u)
            trajectory_log.append(log)
            curr_x = (curr_x - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
            curr_u = (curr_u - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
            hist_xs = torch.cat([hist_xs[:, 1:], torch.from_numpy(curr_x).float().to(DEVICE).reshape(-1, 1, 1)], dim=1)
            hist_us = torch.cat([hist_us[:, 1:], torch.from_numpy(curr_u).float().to(DEVICE).reshape(-1, 1, 1)], dim=1)
            print('Environment Execution Done, Total Time: {}'.format(perf_counter() - start_time))
    trajectory_x = np.stack(trajectory_x)
    trajectory_u = np.stack(trajectory_u)
    return np.split(trajectory_x, state_split_list, axis=1),np.split(trajectory_u, action_split_list,
                                                                      axis=1), trajectory_log


if __name__ == '__main__':
    with open('data/control/prb_config.pkl', 'rb') as f:
        prb_config = pickle.load(f)
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    with open('data/control/target_test.pkl', 'rb') as f:
        target_test = pickle.load(f)

    num_x_list = [4]
    num_heaters_list = [12]
    mpc_config = {
        'ridge_coefficient': prb_config['ridge_coefficient'],
        'smoothness_coefficient': prb_config['smoothness_coefficient'],
        'receding_history': prb_config['receding_history'],
        'receding_horizon': prb_config['receding_horizon'],
        'target_values_list': target_test['target_values_list'],
        'target_times_list': target_test['target_times_list'],
        'max_iter': 200,
        'loss_threshold': 1e-9,
        'opt_config': {'lr': 2e-1},
        'scheduler_config': {'patience': 5, 'factor': 0.5, 'min_lr': 1e-4}
    }
    model_names = ['ICGAT']
    data_ratio_list = [1.0]

    state_pos_list = []
    action_pos_list = []
    graph_list = []
    num_graphs = len(target_test['target_values_list'])

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            if not os.path.exists('design_opt_experiment/test/{}_{}'.format(num_x, num_heaters)):
                os.makedirs('design_opt_experiment/test/{}_{}'.format(num_x, num_heaters))
            for model_name in model_names:
                for data_ratio in data_ratio_list:
                    print('Now Num of x: {}, Num of Heater: {}, Model: {}_{}'.format(num_x, num_heaters, model_name, data_ratio))
                    with open('saved_model/train_config_{}_{}.pkl'.format(model_name, data_ratio), 'rb') as f:
                        model_config = pickle.load(f)
                    with open('design_opt_experiment/train/{}_{}/design_opt_experiment_result_{}_{}.pkl'.format(num_x,
                                                                                                                num_heaters,
                                                                                                                model_name,
                                                                                                                data_ratio),
                              'rb') as f:
                        design_opt_experiment_result = pickle.load(f)
                    graph = design_opt_experiment_result['optimized_graph']
                    x_trajectory_list, u_trajectory_list, log_trajectory_list = run_mpc(mpc_config,
                                                                                        env_config,
                                                                                        model_config,
                                                                                        graph)
                    with open('design_opt_experiment/test/{}_{}/mpc_config_{}_{}.pkl'.format(num_x,
                                                                                             num_heaters,
                                                                                             model_name,
                                                                                             data_ratio),
                              'wb') as f:
                        pickle.dump(mpc_config, f)

                    mpc_experiment_result = {
                        'x_trajectory_list': x_trajectory_list,
                        'u_trajectory_list': u_trajectory_list,
                        'log_trajectory_list': log_trajectory_list
                    }
                    with open('design_opt_experiment/test/{}_{}/mpc_experiment_result_{}_{}.pkl'.format(num_x,
                                                                                                        num_heaters,
                                                                                                        model_name,
                                                                                                        data_ratio),
                              'wb') as f:
                        pickle.dump(mpc_experiment_result, f)
