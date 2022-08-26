from os.path import join
from timeit import default_timer
from datetime import datetime

import os
import pickle

import torch
import wandb
import dgl
from torch.utils.data import TensorDataset, DataLoader

from src.utils.preprocess_data import preprocess_data
from src.model.get_model import get_model
from src.utils.penalty_utils import get_nonnegative_penalty, project_params
from src.utils.fix_seed import fix_seed
fix_seed()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')


def train_model(train_config):
    # Training Parameter Settings
    model_name = train_config['model_name']
    hidden_dim = train_config['hidden_dim']
    is_convex = train_config['is_convex']
    data_ratio = train_config['data_ratio']
    receding_history = train_config['receding_history']
    receding_horizon = train_config['receding_horizon']
    state_scaler = train_config['state_scaler']
    action_scaler = train_config['action_scaler']
    x_dim = train_config['x_dim']
    u_dim = train_config['u_dim']
    decaying_coefficient_list = train_config['decaying_coefficient_list']
    bs = train_config['bs']
    epoch = train_config['epoch']
    save_every = train_config['save_every']
    test_every = train_config['test_every']
    lrs = train_config['lrs']
    # Get Model
    m = get_model(model_name=model_name, hidden_dim=hidden_dim, is_convex=is_convex).to(DEVICE)
    m.train()

    # Data Loading & Pre-processing
    train_data_path = 'data/ground_truth/train_data.pkl'
    test_data_path = 'data/ground_truth/val_data.pkl'
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    train_history_xs_list, train_history_us_list, train_us_list, train_ys_list, train_graph_list, train_idxs = preprocess_data(
        states=train_data['state_list'],
        actions=train_data['action_list'],
        graphs=train_data['graph_list'],
        receding_history=receding_history,
        receding_horizon=receding_horizon,
        state_scaler=state_scaler,
        action_scaler=action_scaler,
        data_ratio=data_ratio,
        device=DEVICE)
    test_history_xs_list, test_history_us_list, test_us_list, test_ys_list, test_graph_list, test_idxs = preprocess_data(
        states=test_data['state_list'],
        actions=test_data['action_list'],
        graphs=test_data['graph_list'],
        receding_history=receding_history,
        receding_horizon=receding_horizon,
        state_scaler=state_scaler,
        action_scaler=action_scaler,
        device=DEVICE)
    with torch.no_grad():
        test_batch_size = test_idxs.shape[0]
        test_gs = dgl.batch([test_graph_list[test_idxs[idx, 0]] for idx in range(test_batch_size)])
        test_history_x = torch.cat([test_history_xs_list[test_idxs[idx, 0]][test_idxs[idx, 1]] for idx in range(test_batch_size)])
        test_history_u = torch.cat([test_history_us_list[test_idxs[idx, 0]][test_idxs[idx, 1]] for idx in range(test_batch_size)])
        test_u = torch.cat([test_us_list[test_idxs[idx, 0]][test_idxs[idx, 1]] for idx in range(test_batch_size)])
        test_y = torch.cat([test_ys_list[test_idxs[idx, 0]][test_idxs[idx, 1]] for idx in range(test_batch_size)])
    train_dataset = TensorDataset(train_idxs)

    run = wandb.init(config=train_config,
                     project='ICGNN',
                     entity='55mong',
                     reinit=True,
                     group='HeatSimulatorIEEE',
                     name='{}_{}'.format(model_name, data_ratio))
    test_best_loss = float('inf')
    now =datetime.now()
    saved_path = now.strftime('saved_model/%Y%m%d%H%M%S')
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    with open(join(saved_path, 'train_config_{}_{}.pkl'.format(model_name, data_ratio)), 'wb') as f:
        pickle.dump(train_config, f)

    for lr in lrs:
        for decaying_coefficient in decaying_coefficient_list:
            print('Now Decaying Coefficient: {}'.format(decaying_coefficient))
            decaying_list = []
            coef = 1.0
            for i in range(receding_horizon):
                decaying_list.append(coef)
                coef = coef * decaying_coefficient
            decaying_list = torch.tensor(decaying_list, device=DEVICE)

            def criteria(y1, y2):
                crit = torch.nn.SmoothL1Loss(reduction='none')
                return torch.einsum('bnk, n->bnk', crit(y1, y2), decaying_list).mean()

            def val_criteria(y1, y2):
                crit = torch.nn.SmoothL1Loss()
                return crit(y1, y2)
            opt = torch.optim.Adam(m.parameters(), lr=lr)

            num_updates = 0
            # Start Training
            for ep in range(epoch):
                print('{} th epoch'.format(ep))
                train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                iters = len(train_dl)
                for i, (train_idx, ) in enumerate(train_dl):
                    start_time = default_timer()
                    batch_size = train_idx.shape[0]
                    # Batching Data
                    gs = dgl.batch([train_graph_list[train_idx[idx, 0]] for idx in range(batch_size)])
                    history_x = torch.cat([train_history_xs_list[train_idx[idx, 0]][train_idx[idx, 1]] for idx in range(batch_size)])
                    history_u = torch.cat([train_history_us_list[train_idx[idx, 0]][train_idx[idx, 1]] for idx in range(batch_size)])
                    u = torch.cat([train_us_list[train_idx[idx, 0]][train_idx[idx, 1]] for idx in range(batch_size)])
                    y = torch.cat([train_ys_list[train_idx[idx, 0]][train_idx[idx, 1]] for idx in range(batch_size)])

                    # Compute the Forward path
                    h0 = m.filter_history(gs, history_x, history_u)
                    predicted_y = m.rollout(gs, h0, u)

                    # Compute the Training loss
                    loss = criteria(predicted_y, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    project_params(m)
                    end_time = default_timer()

                    log_dict = dict()
                    log_dict['Epoch'] = ep + i / iters
                    log_dict['fit_time'] = end_time - start_time
                    log_dict['train_loss'] = loss.item()
                    num_updates += 1
                    if num_updates % test_every == 0:
                        with torch.no_grad():
                            test_h0 = m.filter_history(test_gs, test_history_x, test_history_u)
                            test_predicted_y = m.rollout(test_gs, test_h0, test_u)
                            test_loss = val_criteria(test_predicted_y, test_y)
                            log_dict['test_loss'] = test_loss.item()
                            if test_loss < test_best_loss:
                                test_best_loss = test_loss
                                torch.save(m.state_dict(), join(saved_path, '{}_{}.pt'.format(model_name, data_ratio)))
                            log_dict['test_best_loss'] = test_best_loss
                    if num_updates % save_every == 0:
                        torch.save(m.state_dict(), join(wandb.run.dir, 'model.pt'))
                    wandb.log(log_dict)
    run.finish()


if __name__ == '__main__':
    with open('data/env_config.pkl', 'rb') as f:
        env_config = pickle.load(f)
    data_ratio_list = [1.0]
    model_names = ['GAT', 'ICGAT']
    hidden_dims = [64]
    is_convex = False
    for data_ratio in data_ratio_list:
        for hidden_dim in hidden_dims:
            for model_name in model_names:
                train_config = {
                    'model_name': model_name,
                    'hidden_dim': hidden_dim,
                    'is_convex': is_convex,
                    'data_ratio': data_ratio,
                    'receding_history': 5,
                    'receding_horizon': 10,
                    'state_scaler': env_config['state_scaler'],
                    'action_scaler': env_config['action_scaler'],
                    'x_dim': 1,
                    'u_dim': 1,
                    'decaying_coefficient_list': [0.2, 0.4, 0.6, 0.8, 1.0],
                    'bs': 64,
                    'epoch': int(1000 / data_ratio),
                    'save_every': 25,
                    'test_every': 25,
                    'lrs': [1e-3]
                }
                train_model(train_config)
