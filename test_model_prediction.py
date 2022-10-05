import argparse
import pickle

import dgl
import matplotlib

matplotlib.rcParams['font.family'] = ['Noto Serif', 'Serif']
import matplotlib.pyplot as plt

# plt.style.use('seaborn-talk')
plt.style.use('seaborn-poster')

import numpy as np
import torch
import yaml
from matplotlib import ticker

from src.model.get_model import get_model
from src.utils.preprocess_data import preprocess_data

# plt.style.use('seaborn')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def predict_future(model_names, test_data, data_preprocessing_config, test_config):
    history_len = test_config['history_len']
    receding_horizon = test_config['receding_horizon']

    data_preprocessing_config['history_len'] = history_len
    data_preprocessing_config['receding_horizon'] = receding_horizon
    data_preprocessing_config['device'] = device
    test_hist_xs, test_hist_us, test_future_us, test_future_xs, test_gs, test_idxs = preprocess_data(test_data,
                                                                                                     data_preprocessing_config)
    with torch.no_grad():
        tg = dgl.batch([test_gs[idx[0]] for idx in test_idxs])
        thx = torch.cat([test_hist_xs[idx[0]][idx[1]] for idx in test_idxs])
        thu = torch.cat([test_hist_us[idx[0]][idx[1]] for idx in test_idxs])
        tfu = torch.cat([test_future_us[idx[0]][idx[1]] for idx in test_idxs])
        tfx = torch.cat([test_future_xs[idx[0]][idx[1]] for idx in test_idxs])

    model_test_loss_dict = {}

    def crit(x, y):
        loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        mean = loss_fn(x, y).mean(dim=(0, 2)).detach().cpu().numpy()
        std = loss_fn(x, y).std(dim=(0, 2)).detach().cpu().numpy()
        mean = np.concatenate([[0], mean])
        std = np.concatenate([[0], std])
        return mean, std

    for model_name in model_names:
        model_config = yaml.safe_load(open('config/model/{}/model_config.yaml'.format(model_name), 'r'))
        m = get_model(model_name, model_config, True).to(device)
        with torch.no_grad():
            pfx = m.multistep_prediction(tg, thx, thu, tfu)
            model_test_loss_dict[model_name] = crit(tfx, pfx)
    return model_test_loss_dict


def plot_test_loss(model_loss_dict):
    gamma = 0.1
    label_size = 15
    fig, axes = plt.subplots(1, 1, figsize=(7, 4))
    # fig, axes = plt.subplots(1, 1)
    for model_name in model_loss_dict.keys():
        mean = model_loss_dict[model_name][0]
        std = model_loss_dict[model_name][1]
        axes.plot(mean, label=model_name, linewidth=2)
        axes.fill_between(range(mean.shape[0]),
                          mean - gamma * std,
                          mean + gamma * std, alpha=0.2)
    axes.set_xlabel(r'Rollout Steps', fontsize=label_size)
    axes.set_ylabel(r'Prediction MSE', fontsize=label_size)
    # xes.set_ylim([0.0, 0.002])
    # fmt = lambda x, pos: '{:.1f}'.format(x * 1e4)
    axes.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # axes.tick_params(axis='x')
    # axes.tick_params(axis='y')
    axes.legend(fancybox=True, shadow=True)
    axes.grid(True, which='both', ls='--')
    axes.set_xlim(0, 10)
    # axes.set_yscale('log')
    # fig.tight_layout()
    fig.savefig('pred_results2.pdf', bbox_inches='tight')
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', default=['Linear', 'GNN', 'ICGNN'])
    args = parser.parse_args()

    data_generation_config = yaml.safe_load(open('config/data/data_generation_config.yaml', 'r'))
    data_preprocessing_config = yaml.safe_load(open('config/data/data_preprocessing_config.yaml', 'r'))
    test_data = pickle.load(open(data_generation_config['test_data_saved_path'], 'rb'))
    test_config = yaml.safe_load(open('config/prediction/test_config.yaml', 'r'))
    model_loss_dict = predict_future(args.model_names, test_data, data_preprocessing_config, test_config)
    plot_test_loss(model_loss_dict)
