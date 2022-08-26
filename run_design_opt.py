import os
import pickle

import numpy as np
import torch

from src.utils.get_graph import generate_full_graph
from src.model.get_model import get_model
from src.control.design_opt import DesignOptimizer
from src.utils.get_target import generate_target_trajectory
from src.utils.fix_seed import fix_seed

fix_seed()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if not os.path.exists('design_opt_experiment'):
    os.makedirs('design_opt_experiment')


def run_design_opt(prb_config,
                   design_opt_config,
                   model_config,
                   env_config,
                   graph_config,
                   target_config):
    # Read Problem Setting
    ridge_coefficient = prb_config['ridge_coefficient']
    smoothness_coefficient = prb_config['smoothness_coefficient']
    receding_history = prb_config['receding_history']
    receding_horizon = prb_config['receding_horizon']

    # Set Bilevel Optimization Hyperparameters
    num_targets = design_opt_config['num_targets']
    upper_max_iter = design_opt_config['upper_max_iter']
    lower_max_iter = design_opt_config['lower_max_iter']
    upper_loss_threshold = design_opt_config['upper_loss_threshold']
    lower_loss_threshold = design_opt_config['lower_loss_threshold']
    upper_opt_config = design_opt_config['upper_opt_config']
    upper_scheduler_config = design_opt_config['upper_scheduler_config']
    lower_opt_config = design_opt_config['lower_opt_config']
    lower_scheduler_config = design_opt_config['lower_scheduler_config']

    # Set Model Information
    model_name = model_config['model_name']
    hidden_dim = model_config['hidden_dim']
    is_convex = model_config['is_convex']
    data_ratio = model_config['data_ratio']
    saved_model_path = 'saved_model/{}_{}.pt'.format(model_name, data_ratio)
    model = get_model(model_name, hidden_dim, is_convex, saved_model_path).to(DEVICE)

    # Read Environment Setting
    num_cells = env_config['num_cells']
    dt = env_config['dt']
    epsilon = env_config['epsilon']
    domain_range = env_config['domain_range']
    action_min = env_config['action_min']
    action_max = env_config['action_max']
    state_scaler = env_config['state_scaler']
    action_scaler = env_config['action_scaler']

    state_pos = graph_config['state_pos']
    action_pos = graph_config['action_pos']
    graph = graph_config['graph']
    num_states = state_pos.shape[0]
    num_actions = action_pos.shape[0]

    target_values_list = target_config['target_values_list']
    target_times_list = target_config['target_times_list']

    target_list = []
    for (target_values, target_times) in zip(target_values_list, target_times_list):
        target = np.array(generate_target_trajectory(target_values, target_times)).reshape((1, -1, 1))
        target = np.concatenate([target for _ in range(num_states)], axis=0)
        target_list.append(torch.from_numpy(target).float().to(DEVICE))

    position_min = domain_range[0] + epsilon
    position_max = domain_range[1] - epsilon
    u_min = (action_min - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
    u_max = (action_max - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
    is_logging = True

    optimizer = DesignOptimizer(model=model,
                                graph=graph,
                                target_list=target_list,
                                num_states=num_states,
                                num_actions=num_actions,
                                num_targets=num_targets,
                                receding_history=receding_history,
                                ridge_coefficient=ridge_coefficient,
                                smoothness_coefficient=smoothness_coefficient,
                                position_min=position_min,
                                position_max=position_max,
                                u_min=u_min,
                                u_max=u_max,
                                upper_max_iter=upper_max_iter,
                                lower_max_iter=lower_max_iter,
                                upper_loss_threshold=upper_loss_threshold,
                                lower_loss_threshold=lower_loss_threshold,
                                upper_opt_config=upper_opt_config,
                                upper_scheduler_config=upper_scheduler_config,
                                lower_opt_config=lower_opt_config,
                                lower_scheduler_config=lower_scheduler_config,
                                is_logging=is_logging,
                                device=DEVICE)
    design_opt_log = optimizer.solve()
    prev_graph = graph.to('cpu')
    best_action_pos = torch.from_numpy(design_opt_log['best_position']).float().to('cpu')
    torch_state_pos = torch.from_numpy(state_pos).float().to('cpu')
    opt_graph = generate_full_graph(torch_state_pos, best_action_pos, 'cpu')
    return prev_graph, opt_graph, design_opt_log


if __name__ == '__main__':
    with open('data/control/design_opt/prb_config.pkl', 'rb') as f:
        prb_config = pickle.load(f)
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    with open('data/control/design_opt/target_config.pkl', 'rb') as f:
        target_config = pickle.load(f)

    num_x_list = [3, 4, 5]
    num_heaters_list = [5, 10, 15, 20]
    num_repeats = 10
    design_opt_config = {
        'num_targets': 32,
        'upper_max_iter': 100,
        'lower_max_iter': 200,
        'upper_loss_threshold': 1e-9,
        'lower_loss_threshold': 1e-9,
        'upper_opt_config': {'lr': 2e-1},
        'upper_scheduler_config': {'patience': 5, 'factor': 0.5, 'min_lr': 1e-3},
        'lower_opt_config': {'lr': 2e-1},
        'lower_scheduler_config': {'patience': 5, 'factor': 0.5, 'min_lr': 1e-4}
    }
    model_names = ['ICGAT', 'GAT']
    data_ratio_list = [1.0]

    for num_x in num_x_list:
        for num_heaters in num_heaters_list:
            if not os.path.exists('design_opt_experiment/{}_{}'.format(num_x, num_heaters)):
                os.makedirs('design_opt_experiment/{}_{}'.format(num_x, num_heaters))
            for r in range(num_repeats):
                with open('data/control/design_opt/graph_{}_{}_{}.pkl'.format(num_x, num_heaters, r), 'rb') as f:
                    graph_config = pickle.load(f)
                for model_name in model_names:
                    for data_ratio in data_ratio_list:
                        print('Now Num of x: {}, Num of Heater: {}, Repeat: {}, Model: {}_{}'.format(num_x, num_heaters,
                                                                                                     r, model_name,
                                                                                                     data_ratio))
                        with open('saved_model/train_config_{}_{}.pkl'.format(model_name, data_ratio), 'rb') as f:
                            model_config = pickle.load(f)
                        prev_graph, optimized_graph, design_opt_log = run_design_opt(prb_config,
                                                                                     design_opt_config,
                                                                                     model_config,
                                                                                     env_config,
                                                                                     graph_config,
                                                                                     target_config)
                        design_opt_experiment_result = {
                            'prev_graph': prev_graph,
                            'optimized_graph': optimized_graph,
                            'design_opt_log': design_opt_log
                        }
                        with open('design_opt_experiment/{}_{}/design_opt_experiment_result_{}_{}_{}.pkl'.format(num_x,
                                                                                                                 num_heaters,
                                                                                                                 model_name,
                                                                                                                 data_ratio,
                                                                                                                 r),
                                  'wb') as f:
                            pickle.dump(design_opt_experiment_result, f)
