import dgl.function as fn
import torch
import torch.nn as nn

from dgl.nn.functional import edge_softmax


class HistoryFilterLinear(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim):
        super(HistoryFilterLinear, self).__init__()
        self.h0_generator = nn.Linear(2 + x_dim, hidden_dim)
        self.u2h_enc = nn.Linear(u_dim + 2 * 2 + 1, hidden_dim)
        self.x2h_enc = nn.Linear(x_dim + hidden_dim + 2 * 2 + 1, hidden_dim)
        self.h_updater = nn.Linear(2 + hidden_dim * 3 + x_dim, hidden_dim)

    def forward(self, g, h, x, u):
        s2s = ('state', 's2s', 'state')
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['state'].data['x'] = x
            g.nodes['action'].data['u'] = u
            g.update_all(self.u2h_msg, fn.sum('u2h_msg', 'sum_u'), etype='a2s')
            g.update_all(self.x2h_msg, fn.mean('x2h_msg', 'sum_x'), etype='s2s')
            inp = torch.cat([g.nodes['state'].data['pos'],
                             g.nodes['state'].data['h'],
                             g.nodes['state'].data['sum_u'],
                             g.nodes['state'].data['sum_x'],
                             g.nodes['state'].data['x']], dim=-1)
            return self.h_updater(inp)

    def filter_history(self, g, history_xs, history_us):
        history_xs = history_xs.unbind(dim=1)
        history_us = history_us.unbind(dim=1)
        h = self.h0_generator(torch.cat([g.nodes['state'].data['pos'], history_xs[0]], dim=1))
        for i in range(len(history_us)):
            h = self.forward(g, h, history_xs[i + 1], history_us[i])
        return h

    def u2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_u = edges.src['u']
        inp = torch.cat([src_u, src_pos, dst_pos, edge_dis], dim=-1)
        return {'u2h_msg': self.u2h_enc(inp)}

    def x2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_x = edges.src['x']
        src_h = edges.src['h']
        inp = torch.cat([src_x, src_h, src_pos, dst_pos, edge_dis], dim=-1)
        return {'x2h_msg': self.x2h_enc(inp)}
