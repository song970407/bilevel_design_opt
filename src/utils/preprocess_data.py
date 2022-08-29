import torch

from src.utils.scale_data import minmax_scale


def preprocess_data(data, data_preprocessing_config):
    scale_method = data_preprocessing_config['scale_method']
    scaler = data_preprocessing_config['scaler']
    history_len = data_preprocessing_config['history_len']
    receding_horizon = data_preprocessing_config['receding_horizon']
    state_scaler = data_preprocessing_config['state_scaler']
    action_scaler = data_preprocessing_config['action_scaler']
    device = data_preprocessing_config['device']
    history_xs, history_us, future_us, future_xs, gs, idxs = [], [], [], [], [], []
    for i, episode in enumerate(data):
        hxs = []
        hus = []
        fus = []
        fxs = []
        state = torch.from_numpy(episode['state_trajectory']).float().to(device)
        action = torch.from_numpy(episode['action_trajectory']).float().to(device)
        state = torch.cat([torch.zeros((history_len - 1, state.shape[1]), device=device), state], dim=0)
        action = torch.cat([torch.zeros((history_len - 1, action.shape[1]), device=device), action], dim=0)
        state = state.transpose(0, 1).unsqueeze(dim=-1)
        action = action.transpose(0, 1).unsqueeze(dim=-1)
        if scale_method == 'MinMax':
            state = minmax_scale(state, state_scaler, scaler)
            action = minmax_scale(action, action_scaler, scaler)
        gs.append(episode['graph'].to(device))
        for j in range(state.shape[1] - history_len - receding_horizon + 1):
            hxs.append(state[:, j:j + history_len])
            hus.append(action[:, j:j + history_len - 1])
            fus.append(action[:, j + history_len - 1:j + history_len + receding_horizon - 1])
            fxs.append(state[:, j + history_len:j + history_len + receding_horizon])
            idxs.append(torch.tensor([i, j], device=device))
        history_xs.append(hxs)
        history_us.append(hus)
        future_us.append(fus)
        future_xs.append(fxs)
    idxs = torch.stack(idxs, dim=0).int()
    return history_xs, history_us, future_us, future_xs, gs, idxs
