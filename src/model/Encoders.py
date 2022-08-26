import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax

from src.nn.ConvexNN import PartialConvexNN1
from src.nn.Activation import LearnableLeakyReLU, ConvexPReLU1
from src.nn.NonnegativeLinear import NonnegativeLinear1


class EncoderGCN(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderGCN, self).__init__()
        self.u2h_enc = nn.Sequential(
            nn.Linear(2 * 2 + 1 + u_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc = nn.Sequential(
            nn.Linear(2 * 2 + 1 + hidden_dim, mlp_h_dim),
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
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.u2h_msg, fn.sum('u2h_msg', 'sum_u'), etype='a2s')
            g.update_all(self.h2h_msg, fn.mean('h2h_msg', 'sum_h'), etype='s2s')
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
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_u], dim=-1)
        return {'u2h_msg': self.u2h_enc(inp)}

    def h2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_h], dim=-1)
        return {'h2h_msg': self.h2h_enc(inp)}


class EncoderICGCN(EncoderGCN):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64, is_convex=True):
        super(EncoderICGCN, self).__init__(u_dim, hidden_dim, mlp_h_dim)
        self.u2h_enc = nn.Sequential(
            PartialConvexNN1(2 * 2 + 1, u_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )
        self.h2h_enc = nn.Sequential(
            PartialConvexNN1(2 * 2 + 1, hidden_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )
        self.h_updater = nn.Sequential(
            PartialConvexNN1(2, hidden_dim * 3, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )


class EncoderWeightedGCN(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderWeightedGCN, self).__init__()
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
            nn.Linear(mlp_h_dim, hidden_dim),
            nn.Sigmoid()
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
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.u2h_msg, fn.sum('u2h_msg', 'sum_u'), etype='a2s')
            g.update_all(self.h2h_msg, fn.mean('h2h_msg', 'sum_h'), etype='s2s')
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
        return {'h2h_msg': self.h2h_enc_dis(inp) * self.h2h_enc_h(src_h)}


class EncoderWeightedICGCN(EncoderWeightedGCN):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64, is_convex=True):
        super(EncoderWeightedICGCN, self).__init__(u_dim, hidden_dim, mlp_h_dim)
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
            PartialConvexNN1(2, hidden_dim * 3, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )


class EncoderGAT(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderGAT, self).__init__()
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
            # g[s2s].edata['h2h_attn'] = torch.ones_like(edge_softmax(g[s2s], g[s2s].edata['h2h_logit'])).to(g.nodes['state'].data['pos'].device)

            g.update_all(self.message_func, fn.sum('m', 'sum_h'), etype='s2s')

            # g[s2s].edata['h2h_attn'] = edge_softmax(g[s2s], torch.clamp(g[s2s].edata['h2h_logit'], -1, 1))
            # g.update_all(self.h2h_msg, fn.sum('h2h_logit', 'sum_h'), etype='s2s')

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


class EncoderICGAT(EncoderGAT):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64, is_convex=True):
        super(EncoderICGAT, self).__init__(u_dim, hidden_dim, mlp_h_dim)
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
            PartialConvexNN1(2, hidden_dim * 3, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )


class EncoderClassicGAT(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderClassicGAT, self).__init__()
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
        self.h2h_enc_logit = nn.Sequential(
            nn.Linear(hidden_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc_h = nn.Sequential(
            nn.Linear(2 * 2 + 1 + hidden_dim, mlp_h_dim),
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
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_h], dim=-1)
        h2h_msg = self.h2h_enc_h(inp)
        return {'h2h_logit': self.h2h_enc_logit(h2h_msg), 'h2h_msg': h2h_msg}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['h2h_attn'] * edges.data['h2h_msg']}


class EncoderClassicGAT2(nn.Module):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64):
        super(EncoderClassicGAT2, self).__init__()
        self.u2h_enc_logit = nn.Sequential(
            nn.Linear(2 * 2 + 1 + u_dim + hidden_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.u2h_enc_u = nn.Sequential(
            nn.Linear(2 * 2 + 1 + u_dim + hidden_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc_logit = nn.Sequential(
            nn.Linear(2 * 2 + 1 + hidden_dim + hidden_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc_h = nn.Sequential(
            nn.Linear(2 * 2 + 1 + hidden_dim + hidden_dim, mlp_h_dim),
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
        a2s = ('action', 'a2s', 'state')
        s2s = ('state', 's2s', 'state')
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g[a2s].apply_edges(self.u2h_msg)
            g[a2s].edata['u2h_attn'] = edge_softmax(g[a2s], g[a2s].edata['u2h_logit'])
            g.update_all(self.message_func_u2h, fn.sum('m', 'sum_u'), etype='a2s')
            g[s2s].apply_edges(self.h2h_msg)
            g[s2s].edata['h2h_attn'] = edge_softmax(g[s2s], g[s2s].edata['h2h_logit'])
            g.update_all(self.message_func_h2h, fn.sum('m', 'sum_h'), etype='s2s')
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
        dst_h = edges.dst['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_u, dst_h], dim=-1)
        return {'u2h_logit': self.u2h_enc_logit(inp), 'u2h_msg': self.u2h_enc_u(inp)}

    def h2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_h = edges.src['h']
        dst_h = edges.dst['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_h, dst_h], dim=-1)
        return {'h2h_logit': self.h2h_enc_logit(inp), 'h2h_msg': self.h2h_enc_h(inp)}

    @staticmethod
    def message_func_u2h(edges):
        return {'m': edges.data['u2h_attn'] * edges.data['u2h_msg']}

    @staticmethod
    def message_func_h2h(edges):
        return {'m': edges.data['h2h_attn'] * edges.data['h2h_msg']}


class EncoderClassicICGAT2(EncoderClassicGAT2):
    def __init__(self, u_dim, hidden_dim, mlp_h_dim=64, is_convex=True):
        super(EncoderClassicICGAT2, self).__init__(u_dim, hidden_dim, mlp_h_dim)
        self.u2h_enc_logit = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.u2h_enc_u = nn.Sequential(
            PartialConvexNN1(2 * 2 + 1, u_dim + hidden_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )
        self.h2h_enc_logit = nn.Sequential(
            nn.Linear(2 * 2 + 1, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, mlp_h_dim),
            nn.Tanh(),
            nn.Linear(mlp_h_dim, hidden_dim)
        )
        self.h2h_enc_h = nn.Sequential(
            PartialConvexNN1(2 * 2 + 1, hidden_dim + hidden_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )
        self.h_updater = nn.Sequential(
            PartialConvexNN1(2, hidden_dim * 3, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, mlp_h_dim, mlp_h_dim, True, nn.Tanh(), ConvexPReLU1(mlp_h_dim, is_convex)),
            PartialConvexNN1(mlp_h_dim, mlp_h_dim, 1, hidden_dim, True, None, nn.Identity(), True)
        )

    def u2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_u = edges.src['u']
        dst_h = edges.dst['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_u, dst_h], dim=-1)
        inp_logit = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'u2h_logit': self.u2h_enc_logit(inp_logit), 'u2h_msg': self.u2h_enc_u(inp)}

    def h2h_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_h = edges.src['h']
        dst_h = edges.dst['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_h, dst_h], dim=-1)
        inp_logit = torch.cat([src_pos, dst_pos, edge_dis], dim=-1)
        return {'h2h_logit': self.h2h_enc_logit(inp_logit), 'h2h_msg': self.h2h_enc_h(inp)}
