import torch
import torch.nn as nn
from torch import Tensor

from lib.models.module import Module


class PathEmbedding(Module):
    def __init__(self, num_entities: int, num_relations: int, hidden_dim: int):
        super().__init__()
        self.embed_entities = nn.Embedding(num_entities, hidden_dim)
        self.embed_relations = nn.Embedding(num_relations, hidden_dim)

    def forward(self, paths: Tensor, has_cls: bool = True) -> Tensor:
        if has_cls:
            cls_embedding = self.embed_entities(paths[:, 0])
            paths = paths[:, 1:]

        path_length = paths.shape[1]

        # position of the entity and relation in the path
        entity_idx = Module._get_entity_idx(path_length)
        relation_idx = Module._get_relation_idx(path_length)

        entity_embedding = self.embed_entities(paths[:, entity_idx])
        relation_embedding = self.embed_relations(paths[:, relation_idx])

        path_embedding = [
            entity_embedding[:, pos // 2, :]
            if pos % 2 == 0
            else relation_embedding[:, (pos - 1) // 2, :]
            for pos in range(path_length)]

        if has_cls:
            path_embedding.insert(0, cls_embedding)

        return torch.stack(path_embedding).permute(dims=(1, 0, 2))
