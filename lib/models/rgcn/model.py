from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.models import Model
from lib.models.rgcn import RGCNEmbedding, RGCNLayer
from lib.utils import Graph


class RGCN(Model):
    def __init__(self, num_nodes: int, hidden_dim: int, num_relations: int,
                 num_bases: int = -1,
                 num_hidden_layers: int = 1,
                 dropout: float = 0.0,
                 node_regularization_param: float = 0.0):
        super().__init__()

        self.layers = nn.ModuleList()

        input_layer = RGCNEmbedding(num_nodes, hidden_dim)
        self.layers.append(input_layer)

        for idx in range(num_hidden_layers):
            activation = F.relu if idx < num_hidden_layers - 1 else None
            # num_relations * 2 for inverse relations, but why here?
            layer = RGCNLayer(hidden_dim, hidden_dim, num_relations * 2, num_bases,
                              activation=activation,
                              self_loop=True,
                              dropout=dropout)
            self.layers.append(layer)

        self.reg_param = node_regularization_param
        self.w_relation = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, triplets: Tensor, graph: Graph, **kwargs) -> Tensor:
        super().forward(triplets, graph)

        # entity_emb: num_entities X hidden_dim
        # rel_emb: num_relations X hidden_dim
        node_embedding, edge_embedding = self._encode(graph)

        # batch_size x hidden_dim
        s, r = triplets[:, 0], triplets[:, 1]

        emb_ar = node_embedding[s] * edge_embedding[r]  # batch_size x hidden_dim
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # hidden_dim x batch_size x 1
        emb_c = node_embedding.transpose(0, 1).unsqueeze(1)  # hidden_dim x 1 x num_entities
        out_prod = torch.bmm(emb_ar, emb_c)  # hidden_dim x batch_size x num_entities
        scores = torch.sum(out_prod, dim=0)  # batch_size x num_entities

        # shouldn't sigmoid clip the values and boost performance? it's lowering performance without it
        return torch.sigmoid(scores)

    def loss(self, triplets: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        super().loss(triplets, labels, graph)

        node_embedding, edge_embedding = self._encode(graph)
        scores = self._score_triplet(triplets, node_embedding, edge_embedding)
        predict_loss = F.binary_cross_entropy_with_logits(scores, labels)
        reg_loss = torch.mean(node_embedding.pow(2)) + torch.mean(edge_embedding.pow(2))

        return predict_loss + self.reg_param * reg_loss

    def _encode(self, graph: Graph) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            layer(graph)

        node_embedding = graph.ndata.pop("h")
        edge_embedding = self.w_relation

        return node_embedding, edge_embedding

    def _score_triplet(self, triplets: Tensor, node_embedding: Tensor, edge_embedding: Tensor) -> Tensor:
        s = node_embedding[triplets[:, 0]]
        r = edge_embedding[triplets[:, 1]]
        o = node_embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)

        return score
