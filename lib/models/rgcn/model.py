from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from torch import Tensor

from lib.models import Model
from lib.models.rgcn.embedding import RGCNEmbedding
from lib.utils import Graph
from pdb import set_trace


class RGCN(Model):
    def __init__(self, num_nodes: int, hidden_dim: int, num_relations: int,
                 num_bases: int = -1,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 node_regularization_param: float = 0.0,
                 regularizer: str = "basis"):
        super().__init__()

        self.layers = nn.ModuleList([RGCNEmbedding(num_nodes, hidden_dim)])

        for i in range(num_layers - 1):
            layer = RelGraphConv(in_feat=hidden_dim, out_feat=hidden_dim, num_rels=num_relations,
                                 regularizer=regularizer, num_bases=num_bases, activation=F.relu, self_loop=True,
                                 dropout=dropout)
            self.layers.append(layer)

        final_layer = RelGraphConv(in_feat=hidden_dim, out_feat=hidden_dim, num_rels=num_relations,
                                   regularizer=regularizer, num_bases=num_bases, self_loop=True, dropout=dropout)
        self.layers.append(final_layer)

        self.reg_param = node_regularization_param
        self.w_relation = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, triplets: Tensor, graph: Graph, **kwargs) -> Tensor:
        super().forward(triplets, graph)
        link_predict = kwargs.get("link_predict", False)

        # entity_emb: num_entities X hidden_dim
        # rel_emb: num_relations X hidden_dim
        node_embedding, edge_embedding = self._encode(graph)

        if link_predict:
            # batch_size x hidden_dim
            s, d = triplets[:, 0], triplets[:, 2]

            emb_nodes = node_embedding[s] * node_embedding[d]  # out: batch_size x hidden_dim
            emb_nodes = emb_nodes.transpose(0, 1).unsqueeze(2)  # out: hidden_dim x batch_size x 1
            emb_c = edge_embedding.transpose(0, 1).unsqueeze(1)  # out hidden_dim x 1 x num_relations
            out_prod = torch.bmm(emb_nodes, emb_c)  # out: hidden_dim x batch_size x num_entities
            scores = torch.sum(out_prod, dim=0)  # out: batch_size x num_entities
        else:
            # batch_size x hidden_dim
            s, r = triplets[:, 0], triplets[:, 1]

            emb_ar = node_embedding[s] * edge_embedding[r]  # batch_size x hidden_dim
            emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # hidden_dim x batch_size x 1
            emb_c = node_embedding.transpose(0, 1).unsqueeze(1)  # hidden_dim x 1 x num_entities
            out_prod = torch.bmm(emb_ar, emb_c)  # hidden_dim x batch_size x num_entities
            scores = torch.sum(out_prod, dim=0)  # batch_size x num_entities

        return torch.sigmoid(scores)

    def loss(self, triplets: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        super().loss(triplets, labels, graph)

        node_embedding, edge_embedding = self._encode(graph)
        scores = RGCN._score_triplet(triplets, node_embedding, edge_embedding)
        predict_loss = F.binary_cross_entropy_with_logits(scores, labels)
        reg_loss = torch.mean(node_embedding.pow(2)) + torch.mean(edge_embedding.pow(2))

        return predict_loss + self.reg_param * reg_loss

    def _encode(self, graph: Graph) -> Tuple[Tensor, Tensor]:
        node_embedding, edge_types = None, graph.edata["type"]
        edge_norm = RGCN.get_edge_norm(graph)

        for layer in self.layers:
            node_embedding = layer(graph, node_embedding, edge_types, edge_norm)

        edge_embedding = self.w_relation

        return node_embedding, edge_embedding

    @classmethod
    def _score_triplet(cls, triplets: Tensor, node_embedding: Tensor, edge_embedding: Tensor) -> Tensor:
        s = node_embedding[triplets[:, 0]]
        r = edge_embedding[triplets[:, 1]]
        o = node_embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score

    @classmethod
    def get_edge_norm(cls, graph: Graph) -> Tensor:
        graph = graph.local_var()
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        return graph.edata['norm']
