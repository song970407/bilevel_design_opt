import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from src.nn.MLP import MultiLayerPerceptron as MLP


def get_aggregator(mode, from_field='m', to_field='agg_m'):
    AGGR_TYPES = ['sum', 'mean', 'max']
    if mode in AGGR_TYPES:
        if mode == 'sum':
            aggr = fn.sum(from_field, to_field)
        if mode == 'mean':
            aggr = fn.mean(from_field, to_field)
        if mode == 'max':
            aggr = fn.max(from_field, to_field)
    else:
        raise RuntimeError("Given aggregation mode {} is not supported".format(mode))
    return aggr


class MPNN(nn.Module):
    def __init__(self,
                 node_indim: int,
                 node_outdim: int,
                 edge_outdim: int,  # updated edge feature dimension
                 edge_indim: int = None,
                 node_aggregator: str = 'sum',
                 mlp_params: dict = {}):
        super(MPNN, self).__init__()

        not_use_edge = True if edge_indim is None or edge_outdim is None else False
        self.use_edge = not not_use_edge

        # infer dimensions
        if self.use_edge:
            em_indim = 2 * node_indim + edge_indim
        else:
            em_indim = 2 * node_indim
        em_outdim = edge_outdim
        nm_indim = node_indim + edge_outdim
        nm_outdim = node_outdim

        self.edge_model = MLP(input_dimension=em_indim,
                              output_dimension=em_outdim,
                              **mlp_params)

        # overload attention mlp params
        self.attn_model = MLP(input_dimension=em_indim,
                              output_dimension=1,
                              **mlp_params)

        self.node_model = MLP(input_dimension=nm_indim,
                              output_dimension=nm_outdim,
                              **mlp_params)
        self.node_aggr = get_aggregator(node_aggregator)

    def forward(self, g, nf, ef=None):
        with g.local_scope():
            g.ndata['_h'] = nf
            if ef is not None:
                g.edata['_h'] = ef

            # perform edge update
            g.apply_edges(self.edge_update)
            g.edata['attn'] = dglnn.edge_softmax(g, g.edata['logits'])

            # update nodes
            g.update_all(message_func=self.message_func,
                         reduce_func=self.node_aggr,
                         apply_node_func=self.node_update)
            updated_ef = g.edata['uh']
            updated_nf = g.ndata['uh']
            return updated_nf, updated_ef

    def edge_update(self, edges):
        sender_nf = edges.src['_h']
        receiver_nf = edges.dst['_h']

        if self.use_edge:
            ef = edges.data['_h']
            e_model_input = torch.cat([sender_nf, receiver_nf, ef], dim=-1)
        else:
            e_model_input = torch.cat([sender_nf, receiver_nf], dim=-1)

        uh = self.edge_model(e_model_input)
        logits = self.attn_model(e_model_input)
        return {'uh': uh, 'logits': logits}

    @staticmethod
    def message_func(edges):
        return {'m': edges.data['uh'] * edges.data['attn']}

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['_h']
        nm_input = torch.cat([agg_m, nf], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'uh': updated_nf}
