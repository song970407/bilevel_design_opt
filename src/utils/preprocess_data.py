from typing import List

import torch


def _preprocess_data(states,  # T+1 x state_dim
                     actions,  # T x action_dim
                     graphs,
                     receding_history,  # int
                     receding_horizon,  # int
                     state_scaler=(0.0, 1.0),
                     action_scaler=(0.0, 1.0),
                     data_ratio=1.0,
                     device='cpu'):
    if type(states) != list:
        states = [states]
    if type(actions) != list:
        actions = [actions]
    if type(graphs) != list:
        graphs = [graphs]
    num_dataset = int(len(states) * data_ratio)
    history_xs_list, history_us_list, us_list, ys_list, graph_list = [], [], [], [], []
    idx_list = []
    for idx in range(num_dataset):
        state = states[idx]
        action = actions[idx]
        num_state = state.shape[1]
        num_action = action.shape[1]
        if state.shape[0] == action.shape[0]:
            state = torch.cat([torch.zeros(receding_history, num_state).to(state.device), state], dim=0)
            action = torch.cat([torch.zeros(receding_history - 1, num_action).to(action.device), action], dim=0)
        else:
            state = torch.cat([torch.zeros(receding_history - 1, num_state).to(state.device), state], dim=0)
            action = torch.cat([torch.zeros(receding_history - 1, num_action).to(action.device), action], dim=0)
        state = state.transpose(0, 1).unsqueeze(dim=-1)  # state_dim x T+1 x 1
        action = action.transpose(0, 1).unsqueeze(dim=-1)  # action_dim x T x 1
        state = (state - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
        action = (action - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
        T = state.shape[1]  # length of trajectory
        history_xs, history_us, us, ys = [], [], [], []
        for itr in range(T - receding_history - receding_horizon + 1):
            history_xs.append(state[:, itr:itr + receding_history])
            history_us.append(action[:, itr:itr + receding_history - 1])
            us.append(action[:, itr + receding_history - 1:itr + receding_history + receding_horizon - 1])
            ys.append(state[:, itr + receding_history:itr + receding_history + receding_horizon])
            idx_list.append(torch.tensor([idx, itr]))
        history_xs = torch.stack(history_xs).to(device)  # D x state_dim x receding_history
        history_us = torch.stack(history_us).to(device)  # D x action_dim x receding_history-1
        us = torch.stack(us).to(device)  # D x action_dim x receding_horizon
        ys = torch.stack(ys).to(device)  # D x state_dim x receding_horizon
        history_xs_list.append(history_xs)
        history_us_list.append(history_us)
        us_list.append(us)
        ys_list.append(ys)
        graph_list.append(graphs[idx].to(device))
    idx_torch = torch.stack(idx_list).int().to(device)
    return history_xs_list, history_us_list, us_list, ys_list, graph_list, idx_torch


def preprocess_data(data, data_preprocessing_config):
    scaler = data_preprocessing_config['scaler']
    scale = data_preprocessing_config['scale']
    history_len = data_preprocessing_config['history_len']
    receding_horizon = data_preprocessing_config['receding_horizon']
    state_min = data_preprocessing_config['state_min']
    state_max = data_preprocessing_config['state_max']
    action_min = data_preprocessing_config['action_min']
    action_max = data_preprocessing_config['action_max']
    device = data_preprocessing_config['device']
    index = []
    history_states = []
    history_actions = []
    actions = []
    future_states = []

    # for i, episode in enumerate(data): # Start from here
