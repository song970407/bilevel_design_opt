import dgl.function as fn
import torch
import torch.nn as nn


class SurrogateModel(nn.Module):
    def __init__(self,
                 history_filter: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super(SurrogateModel, self).__init__()
        self.history_filter = history_filter
        self.encoder = encoder
        self.decoder = decoder

    def multistep_prediction(self, g, history_xs, history_us, us):
        """
        :param g: graph
        :param history_xs: history states [#.total state nodes x history_len x state_dim]
        :param history_us: history actions [#.total control nodes x history_len x control_dim]
        :param us: u_t ~ u_t+(k-1)  [#. total control nodes x receding_horizon x control dim]
        :return:
        """
        h = self.history_filter.filter_history(g, history_xs, history_us)
        us = us.unbind(dim=1)
        hs = []
        for u in us:
            h = self.encoder(g, h, u)
            hs.append(h)
        hs = torch.stack(hs, dim=-2)
        xs = self.decoder(hs)
        return xs
