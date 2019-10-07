from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.models import Model
from lib.models.path_transform.embedding import PathEmbedding
from lib.utils import Graph


class LinkPredict(Model):
    def __init__(self, max_hops: int, num_entities: int, num_relations: int,
                 hidden_dim: int,
                 num_att_heads: int,
                 num_transformer_layers: int):
        super().__init__()

        self.max_hops = max_hops
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_PAD = num_entities
        self.relation_PAD = num_relations
        self.extension_PAD = [self.relation_PAD, self.entity_PAD]
        self.hidden_dim = hidden_dim

        # +1 for padding
        self.embed_path = PathEmbedding(num_entities + 1, num_relations + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_att_heads, hidden_dim)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transform = nn.TransformerEncoder(encoder_layer, num_transformer_layers, encoder_norm)

        self.linear_layers = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(1e-3),
            nn.Linear(hidden_dim, num_relations + 1)
        ])

    def forward(self, data: Tensor, graph: Graph, **kwargs) -> Tensor:
        src_node_id, dst_node_id = data.T
        paths = kwargs.get("subgraph")
        mask_tensors = kwargs.get("masks")

        if len(paths) == 0:
            scores = torch.zeros(1, self.num_relations + 1)
            scores[-1] = 1
            return scores

        mask_tensors = mask_tensors.squeeze(0)

        # embed the paths with transformer
        paths = list(map(lambda x: torch.tensor(x, dtype=torch.long), paths))
        # out: batch_size * path_length * hidden_dim
        path_tensors = torch.stack(list(map(self.embed_path, paths)))
        # out: path_length * batch_size * hidden_dim
        path_tensors = path_tensors.permute(1, 0, 2)
        # out: path_length * batch_size * hidden_dim
        path_embedding = self.transform(path_tensors, src_key_padding_mask=mask_tensors)

        avg_path_embedding = path_embedding.mean(dim=0)  # out: batch_size * hidden_dim

        # calculating attention of paths conditioned on the original source and destination node representation
        src_embedding = self.embed_path.embed_entities(src_node_id)  # out: 1 * hidden_dim
        dst_embedding = self.embed_path.embed_entities(dst_node_id)  # out: 1 * hidden_dim

        # out: batch_size
        attention_weights = (src_embedding * avg_path_embedding * dst_embedding).sum(dim=1)
        # out: batch_size
        attention_scores = torch.sigmoid(attention_weights)  # softmax over a large number might hinder training
        # out: path_length * batch_size * hidden_dim
        attended_path_embedding = avg_path_embedding * attention_scores.unsqueeze(1)

        # out: batch_size * hidden_dim
        path_embedding = attended_path_embedding.mean(dim=0)

        # NOTE: unsqueeze(0) because of batch_size=1
        return self.linear_layers(path_embedding).unsqueeze(0)

    def loss(self, data: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        scores = self.forward(data, graph, **kwargs)
        return F.cross_entropy(scores, labels.long())

    def _pad_to_max_length(self, path: Tuple[int], max_length: int) -> Tuple[Tuple[int], Tensor]:
        num_to_pad = (max_length - len(path))
        # divide by 2 because extension consists of 2 elements
        padded_path = path + tuple(self.extension_PAD * (num_to_pad // 2))
        mask = torch.zeros(max_length, dtype=torch.bool)

        if num_to_pad > 0:
            mask[-num_to_pad:] = 1

        return padded_path, mask
