import dgl.function as fn
import torch
import torch.nn as nn

from src.nn.GraphModules.ConvexModule import PartialConvexLinear, PartialConvexLinear3
from src.nn.ReparameterizedLinear import ReparameterizedLinear


class Predictor(nn.Module):

    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(Predictor, self).__init__()

        self.u2h_enc = nn.Sequential(
            nn.Linear(4 + u_dim + h_dim, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

        self.h_updater = nn.Sequential(
            nn.Linear(2 + h_dim * 2 + u_dim, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

    def forward(self, g, h, u):
        with g.local_scope():
            g.ndata['u'] = u
            g.ndata['h'] = h
            g.update_all(self.t2t_msg, fn.sum('u2h_msg', 'sum_h'))

            inp = torch.cat([g.ndata['pos'],
                             g.ndata['h'],
                             g.ndata['sum_h'],
                             g.ndata['u']], dim=-1)
            return self.h_updater(inp)

    def t2t_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        src_u = edges.src['u']
        src_h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, src_u, src_h], dim=-1)
        return {'u2h_msg': self.u2h_enc(inp)}
        # return {'u2h_msg': self.u2h_enc(h)}


class ConvexPredictor(Predictor):
    def __init__(self, u_dim, h_dim, reparam_method='ReLU', mlp_h_dim=32, negative_slope=0.2):
        super(ConvexPredictor, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.u2h_enc = nn.Sequential(
            PartialConvexLinear(2 * 2, u_dim + h_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            #ReparameterizedLinear(h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            #ReparameterizedLinear(mlp_h_dim, h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.h_updater = nn.Sequential(
            PartialConvexLinear(2, h_dim * 2 + u_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )


class HeteroPredictor(nn.Module):
    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(HeteroPredictor, self).__init__()
        self.a2s_enc = nn.Sequential(
            nn.Linear(2 * 2 + 1 + u_dim, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )
        self.s2s_enc = nn.Sequential(
            nn.Linear(2 * 2 + 1 + h_dim, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )
        self.h_updater = nn.Sequential(
            nn.Linear(2 + h_dim * 3, mlp_h_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_h_dim, h_dim),
            nn.Tanh()
        )

    def forward(self, g, h, u):
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.a2s_msg, fn.sum('a2s_msg', 'sum_u'), etype='a2s')
            g.update_all(self.s2s_msg, fn.sum('s2s_msg', 'sum_h'), etype='s2s')
            inp = torch.cat([g.nodes['state'].data['pos'],
                             g.nodes['state'].data['h'],
                             g.nodes['state'].data['sum_u'],
                             g.nodes['state'].data['sum_h']], dim=-1)
            return self.h_updater(inp)

    def a2s_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_u = edges.src['u']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_u], dim=-1)
        return {'a2s_msg': self.a2s_enc(inp)}

    def s2s_msg(self, edges):
        src_pos = edges.src['pos']
        dst_pos = edges.dst['pos']
        edge_dis = edges.data['dis']
        src_h = edges.src['h']
        inp = torch.cat([src_pos, dst_pos, edge_dis, src_h], dim=-1)
        return {'s2s_msg': self.s2s_enc(inp)}


class HeteroPredictor2(HeteroPredictor):
    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(HeteroPredictor2, self).__init__(u_dim, h_dim, mlp_h_dim)

    def forward(self, g, h, u):
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.a2s_msg, fn.sum('a2s_msg', 'sum_u'), etype='a2s')
            g.update_all(self.s2s_msg, fn.mean('s2s_msg', 'sum_h'), etype='s2s')
            inp = torch.cat([g.nodes['state'].data['pos'],
                             g.nodes['state'].data['h'],
                             g.nodes['state'].data['sum_u'],
                             g.nodes['state'].data['sum_h']], dim=-1)
            return self.h_updater(inp)


class HeteroConvexPredictor(HeteroPredictor):
    def __init__(self, u_dim, h_dim, reparam_method='ReLU', mlp_h_dim=32, negative_slope=0.2):
        super(HeteroConvexPredictor, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.a2s_enc = nn.Sequential(
            PartialConvexLinear(2 * 2 + 1, u_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.s2s_enc = nn.Sequential(
            PartialConvexLinear(2 * 2 + 1, h_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.h_updater = nn.Sequential(
            PartialConvexLinear(2, h_dim * 3, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )


class HeteroConvexPredictor3(HeteroPredictor):
    def __init__(self, u_dim, h_dim, reparam_method='ReLU', mlp_h_dim=32, negative_slope=0.2):
        super(HeteroConvexPredictor3, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.a2s_enc = nn.Sequential(
            PartialConvexLinear3(2 * 2 + 1, u_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.s2s_enc = nn.Sequential(
            PartialConvexLinear3(2 * 2 + 1, h_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.h_updater = nn.Sequential(
            PartialConvexLinear3(2, h_dim * 3, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )


class HeteroConvexPredictor4(HeteroPredictor):
    def __init__(self, u_dim, h_dim, reparam_method='ReLU', mlp_h_dim=32, negative_slope=0.2):
        super(HeteroConvexPredictor4, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.a2s_enc = nn.Sequential(
            PartialConvexLinear3(2 * 2 + 1, u_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.s2s_enc = nn.Sequential(
            PartialConvexLinear3(2 * 2 + 1, h_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.h_updater = nn.Sequential(
            PartialConvexLinear3(2, h_dim * 3, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, g, h, u):
        with g.local_scope():
            g.nodes['state'].data['h'] = h
            g.nodes['action'].data['u'] = u
            g.update_all(self.a2s_msg, fn.sum('a2s_msg', 'sum_u'), etype='a2s')
            g.update_all(self.s2s_msg, fn.mean('s2s_msg', 'sum_h'), etype='s2s')
            inp = torch.cat([g.nodes['state'].data['pos'],
                             g.nodes['state'].data['h'],
                             g.nodes['state'].data['sum_u'],
                             g.nodes['state'].data['sum_h']], dim=-1)
            return self.h_updater(inp)


class LinearPredictor(Predictor):
    def __init__(self, u_dim, h_dim, mlp_h_dim=32):
        super(LinearPredictor, self).__init__(u_dim, h_dim, mlp_h_dim)

        self.u2h_enc = nn.Sequential(
            ReparameterizedLinear(u_dim + 6, mlp_h_dim, reparam_method='Softmax'),
            ReparameterizedLinear(mlp_h_dim, h_dim, reparam_method='Softmax'),
        )

        self.h2h_enc = nn.Sequential(
            ReparameterizedLinear(h_dim + 6, mlp_h_dim, reparam_method='Softmax'),
            ReparameterizedLinear(mlp_h_dim, h_dim, reparam_method='Softmax'),
        )

        self.h_updater = nn.Sequential(
            nn.Linear(h_dim * 3, mlp_h_dim),
            nn.Linear(mlp_h_dim, h_dim),
        )


class MonotonePredictor1(HeteroPredictor):
    def __init__(self, u_dim, h_dim, reparam_method='ReLU', mlp_h_dim=32, negative_slope=0.2):
        super(MonotonePredictor1, self).__init__(u_dim, h_dim, mlp_h_dim)
        self.a2s_enc = nn.Sequential(
            PartialConvexLinear3(2 * 2 + 1, u_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.Tanh()
        )
        self.s2s_enc = nn.Sequential(
            PartialConvexLinear3(2 * 2 + 1, h_dim, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.Tanh()
        )
        self.h_updater = nn.Sequential(
            PartialConvexLinear3(2, h_dim * 3, mlp_h_dim, mlp_h_dim, reparam_method=reparam_method),
            nn.LeakyReLU(negative_slope=negative_slope),
            PartialConvexLinear3(mlp_h_dim, mlp_h_dim, h_dim, h_dim, is_end=True, reparam_method=reparam_method),
            nn.Tanh()
        )


class LinearPreCOPredictor(Predictor):

    def __init__(self, u_dim, h_dim):
        super(LinearPreCOPredictor, self).__init__(u_dim, h_dim)
        self.u2h_enc = nn.Linear(u_dim + 6, h_dim)
        self.h2h_enc = nn.Linear(h_dim + 6, h_dim)
        self.h_updater = nn.Linear(h_dim * 3, h_dim)
