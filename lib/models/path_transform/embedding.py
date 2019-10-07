import torch
import torch.nn as nn
from torch import Tensor

from lib.models import Module


class PathEmbedding(Module):
    def __init__(self, num_entities: int, num_relations: int, hidden_dim: int):
        super().__init__()
        self.embed_entities = nn.Embedding(num_entities, hidden_dim)
        self.embed_relations = nn.Embedding(num_relations, hidden_dim)

    def forward(self, path: Tensor) -> Tensor:
        # position of the entity and relation in the path
        entity_idx = Module._get_entity_idx(len(path))
        relation_idx = Module._get_relation_idx(len(path))

        entity_embedding = self.embed_entities(path[entity_idx])
        relation_embedding = self.embed_relations(path[relation_idx])

        path_embedding = torch.stack([
            entity_embedding[pos // 2]
            if pos % 2 == 0
            else relation_embedding[(pos - 1) // 2]
            for pos in range(len(path))])

        return path_embedding
