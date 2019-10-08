from typing import Union

import torch.nn as nn
from dgl.subgraph import DGLSubGraph

from lib.models.module import Module
from lib.utils import Graph


class EntityEmbedding(Module):
    def __init__(self, num_entities: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, hidden_dim)

    def forward(self, graph: Union[Graph, DGLSubGraph]) -> None:
        super().forward(graph)

        entity_ids = graph.ndata['id'].squeeze()
        node_embeddings = self.embedding(entity_ids)
        graph.ndata['h'] = node_embeddings


class RelationEmbedding(Module):
    def __init__(self, num_relations: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_relations, hidden_dim)

    def forward(self, graph: Union[Graph, DGLSubGraph]) -> None:
        super().forward(graph)

        relation_types = graph.edata['type'].squeeze()
        relation_embeddings = self.embedding(relation_types)
        graph.edata['h'] = relation_embeddings
