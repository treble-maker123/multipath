import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.models import Model
from lib.models.path_transform.embedding import PathEmbedding
from lib.utils import Graph


class LinkPredict(Model):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 hidden_dim: int,
                 num_att_heads: int,
                 num_transformer_layers: int):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_PAD = num_entities
        self.relation_PAD = num_relations
        self.extension_PAD = [self.relation_PAD, self.entity_PAD]
        self.hidden_dim = hidden_dim

        # +2 for padding and CLS, + 1 for padding
        self.embed_path = PathEmbedding(num_entities + 2, num_relations + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_att_heads, hidden_dim)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transform = nn.TransformerEncoder(encoder_layer, num_transformer_layers, encoder_norm)

        self.linear_layers = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(1e-3),
            nn.Linear(hidden_dim, num_relations)
        ])

    def forward(self, triplet: Tensor, graph: Graph, **kwargs) -> Tensor:
        src_node_id, _, dst_node_id = triplet.T
        paths = kwargs.get("paths")  # out: num_paths * path-length
        mask_tensors = kwargs.get("masks")  # out: num_paths * path-length
        num_paths = kwargs.get("num_paths")  # out: batch_size * 1

        if len(paths) == 0:
            scores = torch.zeros(1, self.num_relations + 1)
            scores[-1] = 1
            return scores

        # out: batch_size * path_length * hidden_dim
        path_tensors = self.embed_path(paths)
        # out: path_length * batch_size * hidden_dim
        path_tensors = path_tensors.permute(1, 0, 2)
        # out: path_length * batch_size * hidden_dim
        path_embedding = self.transform(path_tensors, src_key_padding_mask=mask_tensors)
        # using the CLS token to represent paths
        path_cls_embedding = path_embedding[0]  # out: batch_size * hidden_dim

        # calculating attention of paths conditioned on the original source and destination node representation
        # src_embedding = self.embed_path.embed_entities(src_node_id)  # out: 1 * hidden_dim
        # dst_embedding = self.embed_path.embed_entities(dst_node_id)  # out: 1 * hidden_dim

        # # out: batch_size
        # attention_weights = (src_embedding * path_cls_embedding * dst_embedding).sum(dim=1)
        # # out: batch_size
        # attention_scores = torch.sigmoid(attention_weights)
        # # out: batch_size * hidden_dim
        # attended_path_embedding = path_cls_embedding * attention_scores.unsqueeze(1)

        # summarize multiple paths into one path embedding
        summarized_path_embeddings = []
        for num_path in num_paths:
            # triplet_embedding = attended_path_embedding[:num_path]

            triplet_embedding = path_cls_embedding[:num_path]
            summarized_path_embeddings.append(triplet_embedding.mean(dim=0))
            path_cls_embedding = path_cls_embedding[num_path:]

            # attended_path_embedding = attended_path_embedding[num_path:]

        # out: batch_size * hidden_dim
        summarized_path_embeddings = torch.stack(summarized_path_embeddings)

        # out: batch_size * num_classes
        output = self.linear_layers(summarized_path_embeddings)

        return output

    def loss(self, triplet: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        scores = self.forward(triplet, graph, **kwargs)
        labels = labels.view(-1)
        return F.cross_entropy(scores, labels.long())
