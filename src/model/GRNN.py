import dgl.function as fn
import torch
import torch.nn as nn


class GRNN(nn.Module):
    def __init__(self,
                 history_filter: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module):
        super(GRNN, self).__init__()
        self.history_filter = history_filter
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, g, h, u):
        return self.encoder(g, h, u)

    def decode(self, x):
        return self.decoder(x)

    def filter_history(self, g, history_xs, history_us):
        """
        Estimate hidden from the sequences of the observations and controls.
        :param g: graph
        :param history_xs: [#.total state nodes x history_len x state_dim]
        :param history_us: [#.total control nodes x (history_len-1) x control_dim]
        :return: filtered history h, [#.total state nodes x hidden_dim]
        """
        return self.history_filter.filter_history(g, history_xs, history_us)

    def rollout(self, g, h, us):
        """
        :param g: graph
        :param h: filtered hidden [#.total state nodes x hidden_dim]
        :param us: u_t ~ u_t+(k-1)  [#. total control nodes x rollout length x control dim]
        :return:
        """
        us = us.unbind(dim=1)
        hs = []
        for u in us:
            h = self.encode(g, h, u)
            hs.append(h)
        hs = torch.stack(hs, dim=-2)
        xs = self.decode(hs)
        return xs

    def multi_step_prediction(self, g, h, us):
        """
        :param g: graph
        :param h: filtered hidden [#.total state nodes x hidden_dim]
        :param us: u_t ~ u_t+(k-1)  [#. total control nodes x rollout length x control dim]
        :return:
        """
        return self.rollout(g, h, us)
