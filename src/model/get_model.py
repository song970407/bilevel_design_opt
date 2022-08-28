import torch

from src.model.HistoryFilters import *
from src.model.Encoders import *
from src.model.Decoders import *
from src.model.SurrogateModel import GRNN


def get_gcn_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterGCN(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderGCN(u_dim=1, hidden_dim=hidden_dim)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=1)
    m = GRNN(history_filter, encoder, decoder)
    return m


def get_weighted_gcn_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterWeightedGCN(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderWeightedGCN(u_dim=1, hidden_dim=hidden_dim)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=1)
    m = GRNN(history_filter, encoder, decoder)
    return m


def get_gat_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterGAT(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderGAT(u_dim=1, hidden_dim=hidden_dim)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=1)
    m = GRNN(history_filter, encoder, decoder)
    return m


def get_classic_gat_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterClassicGAT(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderClassicGAT(u_dim=1, hidden_dim=hidden_dim)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=1)
    m = GRNN(history_filter, encoder, decoder)
    return m


def get_classic_gat2_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterClassicGAT2(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderClassicGAT2(u_dim=1, hidden_dim=hidden_dim)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=1)
    m = GRNN(history_filter, encoder, decoder)
    return m



def get_icgcn_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterGCN(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderICGCN(u_dim=1, hidden_dim=hidden_dim, is_convex=is_convex)
    decoder = ConvexDecoder(hidden_dim=hidden_dim, output_dim=1, is_convex=is_convex)
    m = GRNN(history_filter, encoder, decoder)
    return m


def get_weighted_icgcn_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterWeightedGCN(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderWeightedICGCN(u_dim=1, hidden_dim=hidden_dim, is_convex=is_convex)
    decoder = ConvexDecoder(hidden_dim=hidden_dim, output_dim=1, is_convex=is_convex)
    m = GRNN(history_filter, encoder, decoder)
    return m



def get_icgat_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterGAT(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderICGAT(u_dim=1, hidden_dim=hidden_dim, is_convex=is_convex)
    decoder = ConvexDecoder(hidden_dim=hidden_dim, output_dim=1, is_convex=is_convex)
    m = GRNN(history_filter, encoder, decoder)
    return m


def get_classic_icgat2_model(hidden_dim=16, is_convex=True):
    history_filter = HistoryFilterClassicGAT2(x_dim=1, u_dim=1, hidden_dim=hidden_dim)
    encoder = EncoderClassicICGAT2(u_dim=1, hidden_dim=hidden_dim, is_convex=is_convex)
    decoder = ConvexDecoder(hidden_dim=hidden_dim, output_dim=1, is_convex=is_convex)
    m = GRNN(history_filter, encoder, decoder)
    return m


model_func_dict = {
    'GCN': get_gcn_model,
    'WeightedGCN': get_weighted_gcn_model,
    'GAT': get_gat_model,
    'ClassicGAT': get_classic_gat_model,
    'ClassicGAT2': get_classic_gat2_model,
    'ICGCN': get_icgcn_model,
    'WeightedICGCN': get_weighted_icgcn_model,
    'ICGAT': get_icgat_model,
    'ClassicICGAT2': get_classic_icgat2_model
}


def get_model(model_name, hidden_dim=16, is_convex=True, saved_model_path=None):
    m = model_func_dict[model_name](hidden_dim, is_convex)
    if saved_model_path is not None:
        m.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
    return m
