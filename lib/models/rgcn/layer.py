from pdb import set_trace

import dgl.function as fn
import torch
import torch.nn as nn

from lib.models import Module
from lib.utils import Graph


class RGCNLayer(Module):
    def __init__(self, in_feat: int, out_feat: int, num_relations: int, num_bases: int,
                 activation=None,
                 bias: bool = False,
                 self_loop: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.out_feat = out_feat
        self.activation = activation
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        weight = torch.Tensor(num_relations, num_bases * self.submat_in * self.submat_out)
        self.weight = nn.Parameter(weight)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        self.bias = None
        self.loop_weight = None
        self.dropout = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        if self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, graph: Graph):
        super().forward()

        loop_message = None

        if self.loop_weight is not None:
            loop_message = torch.mm(graph.ndata["h"], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(graph)

        nodes = graph.ndata['h']

        if self.bias is not None:
            nodes = nodes + self.bias
        if self.loop_weight is not None:
            nodes = nodes + loop_message
        if self.activation:
            nodes = self.activation(nodes)

        graph.ndata['h'] = nodes

    def message_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}
