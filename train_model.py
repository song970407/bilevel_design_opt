import os
import yaml
import pickle
import argparse

from os.path import join
from time import perf_counter

import torch
import wandb
import dgl
from torch.utils.data import TensorDataset, DataLoader

from src.utils.preprocess_data import preprocess_data
from src.utils.generate_decaying_coefficient import generate_decaying_coefficient
from src.model.get_model import get_model
from src.utils.penalty_utils import get_nonnegative_penalty, project_params
from src.utils.fix_seed import fix_seed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_model(model_name, train_data, val_data, data_preprocessing_config):
    if not os.path.exists('saved_model/{}'.format(model_name)):
        os.makedirs('saved_model/{}'.format(model_name))
    model_config = yaml.safe_load(open('config/model/{}/model_config.yaml'.format(model_name), 'r'))
    train_config = yaml.safe_load(open('config/model/{}/train_config.yaml'.format(model_name), 'r'))

    model_saved_path = model_config['model_saved_path']
    history_len = train_config['history_len']
    receding_horizon = train_config['receding_horizon']
    decaying_gammas = train_config['decaying_gammas']
    epoch = train_config['epoch']
    bs = train_config['bs']
    test_every = train_config['test_every']
    opt_name = train_config['opt_name']
    opt_config = train_config['opt_config']
    seed_num = train_config['seed_num']
    fix_seed(seed_num)

    data_preprocessing_config['history_len'] = history_len
    data_preprocessing_config['receding_horizon'] = receding_horizon
    data_preprocessing_config['device'] = device

    train_hist_xs, train_hist_us, train_future_us, train_future_xs, train_gs, train_idxs = preprocess_data(train_data,
                                                                                                           data_preprocessing_config)
    val_hist_xs, val_hist_us, val_future_us, val_future_xs, val_gs, val_idxs = preprocess_data(val_data,
                                                                                               data_preprocessing_config)
    with torch.no_grad():
        vg = dgl.batch([val_gs[idx[0]] for idx in val_idxs])
        vhx = torch.cat([val_hist_xs[idx[0]][idx[1]] for idx in val_idxs])
        vhu = torch.cat([val_hist_us[idx[0]][idx[1]] for idx in val_idxs])
        vfu = torch.cat([val_future_us[idx[0]][idx[1]] for idx in val_idxs])
        vfx = torch.cat([val_future_xs[idx[0]][idx[1]] for idx in val_idxs])
    train_dataset = TensorDataset(train_idxs)
    train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    iters = len(train_dl)

    m = get_model(model_name, model_config).to(device)
    m.train()

    config = {
        'model_name': model_name,
        'model_config': model_config,
        'train_config': train_config,
        'data': {
            'train_data': train_data,
            'val_data': val_data,
            'data_preprocessing_config': data_preprocessing_config
        }
    }
    """run = wandb.init(config=config,
                     project='Bilevel_Design_Opt',
                     entity='55mong',
                     reinit=True,
                     name=model_name)"""

    num_updates = 0
    val_best_loss = float('inf')
    val_crit = torch.nn.SmoothL1Loss()

    for decaying_gamma in decaying_gammas:
        opt = getattr(torch.optim, opt_name)(m.parameters(), lr=opt_config['lr'])
        decaying_coefficient = generate_decaying_coefficient(decaying_gamma, receding_horizon, device)

        def crit(x, y):
            loss_fn = torch.nn.SmoothL1Loss(reduction='none')
            loss = loss_fn(x, y).mean(dim=(0, 2))
            return (decaying_coefficient * loss).mean()

        for ep in range(epoch):
            if ep % 10 == 0:
                print('Epoch [{}] / [{}]'.format(ep, epoch))
            for i, (train_idx,) in enumerate(train_dl):
                start_time = perf_counter()
                tg = dgl.batch([train_gs[idx[0]] for idx in train_idx])
                thx = torch.cat([train_hist_xs[idx[0]][idx[1]] for idx in train_idx])
                thu = torch.cat([train_hist_us[idx[0]][idx[1]] for idx in train_idx])
                tfu = torch.cat([train_future_us[idx[0]][idx[1]] for idx in train_idx])
                tfx = torch.cat([train_future_xs[idx[0]][idx[1]] for idx in train_idx])
                pfx = m.multistep_prediction(tg, thx, thu, tfu)
                train_loss = crit(tfx, pfx)
                opt.zero_grad()
                train_loss.backward()
                opt.step()
                project_params(m)
                num_updates += 1
                log = {
                    'Epoch': ep + i / iters,
                    'fit_time': perf_counter() - start_time,
                    'train_loss': train_loss.item()
                }
                if num_updates % test_every == 0:
                    with torch.no_grad():
                        val_loss = val_crit(m.multistep_prediction(vg, vhx, vhu, vfu), vfx)
                    if val_loss.item() < val_best_loss:
                        val_best_loss = val_loss.item()
                        torch.save(m.state_dict(), model_saved_path)
                    # torch.save(m.state_dict(), join(wandb.run.dir, 'model.pt'))
                    log['val_loss'] = val_loss.item()
                    log['val_best_loss'] = val_best_loss
                # wandb.log(log)
    # run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Linear')
    args = parser.parse_args()

    data_generation_config = yaml.safe_load(open('config/data/data_generation_config.yaml', 'r'))
    data_preprocessing_config = yaml.safe_load(open('config/data/data_preprocessing_config.yaml', 'r'))
    train_data = pickle.load(open(data_generation_config['train_data_saved_path'], 'rb'))
    val_data = pickle.load(open(data_generation_config['val_data_saved_path'], 'rb'))
    train_model(args.model_name, train_data, val_data, data_preprocessing_config)
