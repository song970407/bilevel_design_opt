import dgl.function as fn
import torch
import torch.nn as nn

from dgl.nn.functional import edge_softmax


class HistoryFilterGNN(nn.Module):
    def __init__(self, x_dim, u_dim, hidden_dim, mlp_h_dim=64):
        super(HistoryFilterGNN, self).__init__()
        self.h0_generator = nn.Sequential(
            nn.Linear(2 + x_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.u2h_enc_dis = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.u2h_enc_u = nn.Sequential(
            nn.Linear(u_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.x2h_enc_dis = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.x2h_enc_x = nn.Sequential(
            nn.Linear(x_dim + hidden_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h_updater = nn.Sequential(
            nn.Linear(2 + hidden_dim * 3 + x_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )

    def forward(self, g, h, x, u):
        s2s = ('state', 's2s', 'state')
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['state'].data['x'] = x
            g.nodes['action'].data['u'] = u
            g.update_all(self.u2h_msg, fn.sum('u2h_msg', 'sum_u'), etype='a2s')
            g[s2s].apply_edges(self.x2h_msg)
            g[s2s].edata['x2h_attn'] = edge_softmax(g[s2s], g[s2s].edata['x2h_logit'])
            g.update_all(self.message_func, fn.sum('m', 'sum_x'), etype='s2s')
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
        inp = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'u2h_msg': self.u2h_enc_dis(inp) * self.u2h_enc_u(src_u)}

    def x2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_x = edges.src['x']
        src_h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        xh = torch.cat([src_x, src_h], dim=-1)
        return {'x2h_logit': self.x2h_enc_dis(inp), 'x2h_msg': self.x2h_enc_x(xh)}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['x2h_attn'] * edges.data['x2h_msg']}
