import torch
import numpy as np

from typing import Union, List



def load_data(states,  # T+1 x state_dim
              actions,  # T x action_dim
              receding_history,  # int
              receding_horizon,  # int
              state_scaler=(0.0, 1.0),
              action_scaler=(0.0, 1.0),
              device='cpu'):
    if type(states) != list:
        states = [states]
    if type(actions) != list:
        actions = [actions]
    num_dataset = len(states)
    history_xs, history_us, us, ys, graph_idxs = [], [], [], [], []
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
        state = state.transpose(0, 1)  # state_dim x T+1
        action = action.transpose(0, 1)  # action_dim x T
        state = (state - state_scaler[0]) / (state_scaler[1] - state_scaler[0])
        action = (action - action_scaler[0]) / (action_scaler[1] - action_scaler[0])
        T = state.shape[1]  # length of trajectory
        for itr in range(T - receding_history - receding_horizon + 1):
            history_xs.append(state[:, itr:itr + receding_history])
            history_us.append(action[:, itr:itr + receding_history - 1])
            us.append(action[:, itr + receding_history - 1:itr + receding_history + receding_horizon - 1])
            ys.append(state[:, itr + receding_history:itr + receding_history + receding_horizon])
            graph_idxs.append(torch.ones(1) * idx)  # Should be fix if different graphs
    history_xs = torch.stack(history_xs).to(device)  # D x state_dim x receding_history
    history_us = torch.stack(history_us).to(device)  # D x action_dim x receding_history-1
    us = torch.stack(us).to(device)  # D x action_dim x receding_horizon
    ys = torch.stack(ys).to(device)  # D x state_dim x receding_horizon
    graph_idxs = torch.cat(graph_idxs).int().to(device)
    return history_xs, history_us, us, ys, graph_idxs
