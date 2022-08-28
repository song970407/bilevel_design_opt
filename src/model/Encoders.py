import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax

from src.nn.ConvexNN import PartialConvexNN1
from src.nn.Activation import LearnableLeakyReLU, ConvexPReLU1
from src.nn.NonnegativeLinear import NonnegativeLinear1


class EncoderLinear(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderLinear, self).__init__()
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.u2h_enc_dis = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.u2h_enc_u = nn.Linear(u_dim, hidden_dim)
        self.h2h_enc_dis = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc_h = nn.Linear(hidden_dim, hidden_dim)
        self.h_updater_coef = nn.Sequential(
            nn.Linear(2, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, 3 * hidden_dim * hidden_dim)
        )
        self.h_updater_const = nn.Sequential(
            nn.Linear(2, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )

    def forward(self, g, h, u):
        s2s = ('state', 's2s', 'state')
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.u2h_msg, fn.sum('u2h_msg', 'sum_u'), etype='a2s')
            g[s2s].apply_edges(self.h2h_msg)
            g[s2s].edata['h2h_attn'] = edge_softmax(g[s2s], g[s2s].edata['h2h_logit'])
            g.update_all(self.message_func, fn.sum('m', 'sum_h'), etype='s2s')
            pos = g.nodes['state'].data['pos']
            inp = torch.cat([g.nodes['state'].data['h'],
                             g.nodes['state'].data['sum_u'],
                             g.nodes['state'].data['sum_h']], dim=-1)
            coef = self.h_updater_coef(pos).reshape((-1, self.hidden_dim, 3 * self.hidden_dim))
            return torch.einsum('boi,bi -> bo', coef, inp) + self.h_updater_const(pos)

    def u2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_u = edges.src['u']
        inp = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'u2h_msg': self.u2h_enc_dis(inp) * self.u2h_enc_u(src_u)}

    def h2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'h2h_logit': self.h2h_enc_dis(inp), 'h2h_msg': self.h2h_enc_h(src_h)}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['h2h_attn'] * edges.data['h2h_msg']}


class EncoderGNN(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderGNN, self).__init__()
        self.u2h_enc_dis = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.u2h_enc_u = nn.Sequential(
            nn.Linear(u_dim, mlp_h_dim, bias=False),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim, bias=False),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim, bias=False)
        )
        self.h2h_enc_dis = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc_h = nn.Sequential(
            nn.Linear(hidden_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h_updater = nn.Sequential(
            nn.Linear(2 + hidden_dim * 3, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )

    def forward(self, g, h, u):
        s2s = ('state', 's2s', 'state')
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.u2h_msg, fn.sum('u2h_msg', 'sum_u'), etype='a2s')
            g[s2s].apply_edges(self.h2h_msg)
            g[s2s].edata['h2h_attn'] = edge_softmax(g[s2s], g[s2s].edata['h2h_logit'])
            g.update_all(self.message_func, fn.sum('m', 'sum_h'), etype='s2s')
            inp = torch.cat([g.nodes['state'].data['pos'],
                             g.nodes['state'].data['h'],
                             g.nodes['state'].data['sum_u'],
                             g.nodes['state'].data['sum_h']], dim=-1)
            return self.h_updater(inp)

    def u2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_u = edges.src['u']
        inp = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'u2h_msg': self.u2h_enc_dis(inp) * self.u2h_enc_u(src_u)}

    def h2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'h2h_logit': self.h2h_enc_dis(inp), 'h2h_msg': self.h2h_enc_h(src_h)}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['h2h_attn'] * edges.data['h2h_msg']}


class EncoderICGNN(EncoderGNN):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64, is_convex=True):
        super(EncoderICGNN, self).__init__(u_dim, hidden_dim, mlp_h_dim)
        self.u2h_enc_u = nn.Sequential(
            NonnegativeLinear1(u_dim, mlp_h_dim, bias=False),
            ConvexPReLU1(mlp_h_dim, is_convex),
            NonnegativeLinear1(mlp_h_dim, mlp_h_dim, bias=False),
            ConvexPReLU1(mlp_h_dim, is_convex),
            NonnegativeLinear1(mlp_h_dim, hidden_dim, bias=False)
        )
        self.h2h_enc_h = nn.Sequential(
            NonnegativeLinear1(hidden_dim, mlp_h_dim),
            ConvexPReLU1(mlp_h_dim, is_convex),
            NonnegativeLinear1(mlp_h_dim, mlp_h_dim),
            ConvexPReLU1(mlp_h_dim, is_convex),
            NonnegativeLinear1(mlp_h_dim, hidden_dim)
        )
        self.h_updater = nn.Sequential(
            PartialConvexNN1(2, hidden_dim * 3, mlp_h_dim, mlp_h_dim, True, nn.Tanh(),
                             ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(),
                             ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )
