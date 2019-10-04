import torch.nn as nn
from torch import Tensor

from lib.models import Module


class EntityEmbedding(Module):
    def __init__(self, num_entities, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, hidden_dim)

    def forward(self, entity_ids: Tensor) -> Tensor:
        return self.embedding(entity_ids)


class RelationEmbedding(Module):
    def __init__(self, num_relations: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_relations, hidden_dim)

    def forward(self, relation_ids: Tensor) -> Tensor:
        return self.embedding(relation_ids)
