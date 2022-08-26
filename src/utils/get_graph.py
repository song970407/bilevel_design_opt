import dgl
import numpy as np
import torch
from scipy.spatial.distance import cdist


def generate_graph(state_pos, action_pos, s2s_threshold, a2s_threshold, device='cpu'):
    """
    :param state_pos:
    :param action_pos:
    :param s2s_threshold:
    :param a2s_threshold:
    :param device:
    :return:
    """
    action_state_src = []
    action_state_dst = []
    state_state_src = []
    state_state_dst = []
    num_state = state_pos.shape[0]
    num_action = action_pos.shape[0]
    for i in range(num_state):
        for j in range(num_state):
            if 0 < torch.norm(state_pos[i] - state_pos[j], p=2) < s2s_threshold:
                state_state_src.append(i)
                state_state_dst.append(j)
    for i in range(num_action):
        for j in range(num_state):
            if torch.norm(action_pos[i] - state_pos[j], p=2) < a2s_threshold:
                action_state_src.append(i)
                action_state_dst.append(j)
    state_state_src = torch.tensor(state_state_src).int().to(device)
    state_state_dst = torch.tensor(state_state_dst).int().to(device)
    action_state_src = torch.tensor(action_state_src).int().to(device)
    action_state_dst = torch.tensor(action_state_dst).int().to(device)
    s2s_pos1 = state_pos[state_state_src.long()]
    s2s_pos2 = state_pos[state_state_dst.long()]
    a2s_pos1 = action_pos[action_state_src.long()]
    a2s_pos2 = state_pos[action_state_dst.long()]
    s2s_dis = torch.norm(s2s_pos1 - s2s_pos2, p=2, dim=1).unsqueeze(dim=-1)
    a2s_dis = torch.norm(a2s_pos1 - a2s_pos2, p=2, dim=1).unsqueeze(dim=-1)

    data_dict = {
        ('state', 's2s', 'state'): (state_state_src, state_state_dst),
        ('action', 'a2s', 'state'): (action_state_src, action_state_dst)
    }
    num_nodes_dict = {
        'state': num_state,
        'action': num_action
    }
    g = dgl.heterograph(data_dict, num_nodes_dict).to(device)
    g.ndata['pos'] = {'state': state_pos, 'action': action_pos}
    g.edata['dis'] = {'s2s': s2s_dis, 'a2s': a2s_dis}
    return g


def generate_full_graph(state_pos, action_pos, device='cpu'):
    """
    :param state_pos:
    :param action_pos:
    :param s2s_threshold:
    :param a2s_threshold:
    :param device:
    :return:
    """
    action_state_src = []
    action_state_dst = []
    state_state_src = []
    state_state_dst = []
    num_state = state_pos.shape[0]
    num_action = action_pos.shape[0]
    for i in range(num_state):
        for j in range(num_state):
            state_state_src.append(i)
            state_state_dst.append(j)
    for i in range(num_action):
        for j in range(num_state):
            action_state_src.append(i)
            action_state_dst.append(j)
    state_state_src = torch.tensor(state_state_src).int().to(device)
    state_state_dst = torch.tensor(state_state_dst).int().to(device)
    action_state_src = torch.tensor(action_state_src).int().to(device)
    action_state_dst = torch.tensor(action_state_dst).int().to(device)
    s2s_pos1 = state_pos[state_state_src.long()]
    s2s_pos2 = state_pos[state_state_dst.long()]
    a2s_pos1 = action_pos[action_state_src.long()]
    a2s_pos2 = state_pos[action_state_dst.long()]
    # s2s_dis = torch.norm(s2s_pos1 - s2s_pos2, p=2, dim=1).unsqueeze(dim=-1)
    # a2s_dis = torch.norm(a2s_pos1 - a2s_pos2, p=2, dim=1).unsqueeze(dim=-1)
    s2s_dis = torch.square(s2s_pos1 - s2s_pos2).sum(dim=1).unsqueeze(dim=-1)
    a2s_dis = torch.square(a2s_pos1 - a2s_pos2).sum(dim=1).unsqueeze(dim=-1)

    data_dict = {
        ('state', 's2s', 'state'): (state_state_src, state_state_dst),
        ('action', 'a2s', 'state'): (action_state_src, action_state_dst)
    }
    num_nodes_dict = {
        'state': num_state,
        'action': num_action
    }
    g = dgl.heterograph(data_dict, num_nodes_dict).to(device)
    g.ndata['pos'] = {'state': state_pos, 'action': action_pos}
    g.edata['dis'] = {'s2s': s2s_dis, 'a2s': a2s_dis}
    return g
