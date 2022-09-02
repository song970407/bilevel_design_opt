import torch

from src.model.history_filter.Linear import HistoryFilterLinear
from src.model.history_filter.GNN import HistoryFilterGNN
from src.model.encoder.ICGNN import EncoderICGNN
from src.model.encoder.GNN import EncoderGNN
from src.model.encoder.Linear import EncoderLinear
from src.model.decoder.ICGNN import DecoderConvexNN
from src.model.decoder.GNN import DecoderNN
from src.model.decoder.Linear import DecoderLinear
from src.model.SurrogateModel import SurrogateModel


def get_linear_model(model_config):
    hidden_dim = model_config['hidden_dim']
    history_filter = HistoryFilterLinear(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderLinear(u_dim=1, hidden_dim=hidden_dim)
    decoder = DecoderLinear(hidden_dim=hidden_dim, output_dim=1)
    m = SurrogateModel(history_filter, encoder, decoder)
    return m


def get_gnn_model(model_config):
    hidden_dim = model_config['hidden_dim']
    history_filter = HistoryFilterGNN(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderGNN(u_dim=1, hidden_dim=hidden_dim)
    decoder = DecoderNN(hidden_dim=hidden_dim, output_dim=1)
    m = SurrogateModel(history_filter, encoder, decoder)
    return m


def get_icgnn_model(model_config):
    hidden_dim = model_config['hidden_dim']
    is_convex = model_config['is_convex']
    history_filter = HistoryFilterGNN(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderICGNN(u_dim=1, hidden_dim=hidden_dim, is_convex=is_convex)b
    decoder = DecoderConvexNN(hidden_dim=hidden_dim, output_dim=1, is_convex=is_convex)
    m = SurrogateModel(history_filter, encoder, decoder)
    return m


model_name_dict = {
    'Linear': get_linear_model,
    'GNN': get_gnn_model,
    'ICGNN': get_icgnn_model
}


def get_model(model_name, model_config, load_saved_model=False):
    m = model_name_dict[model_name](model_config)
    if load_saved_model:
        m.load_state_dict(torch.load(model_config['model_saved_path'], map_location='cpu'))
    return m
