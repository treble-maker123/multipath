import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pdb import set_trace

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

    @property
    def relation_weights(self) -> torch.Tensor:
        return self.embed_path.embed_relations.weight[:-1]  # [:-1] to exclude padding token

    def forward(self, triplet: Tensor, _: Graph, **kwargs) -> Tensor:
        paths = kwargs.get("paths")  # out: num_paths * path-length
        masks = kwargs.get("masks")  # out: num_paths * path-length
        num_paths = kwargs.get("num_paths")  # out: batch_size * 1

        # out: batch_size * hidden_dim
        path_embeddings = self._encode_paths(paths, masks, num_paths)
        # out: hidden_dim, num_rels
        rel_weights = self.relation_weights.permute(1, 0)

        # out: batch_size * num_rels
        scores = torch.mm(path_embeddings, rel_weights)

        return scores

        # return self._encode_paths(paths, masks, num_paths)

    def loss(self, triplet: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        paths = kwargs.get("paths")  # out: num_paths * path-length
        masks = kwargs.get("masks")  # out: num_paths * path-length
        num_paths = kwargs.get("num_paths")  # out: batch_size * 1

        target_rel_idx = triplet[:, 1]
        labels = labels.view(-1)

        # num_rels * hidden_dim
        rel_weights = self.relation_weights
        # out: batch_size * hidden_dim
        path_embeddings = self._encode_paths(paths, masks, num_paths)
        # out: batch_size * hidden_dim
        target_rel_weights = rel_weights[target_rel_idx]

        # out: batch_size
        scores = (path_embeddings * target_rel_weights).sum(dim=1)
        scores = torch.sigmoid(scores)

        return F.binary_cross_entropy_with_logits(scores, labels.float())

        # hackathon formulation
        # scores = self._encode_paths(paths, masks, num_paths)
        # return F.cross_entropy(scores, labels)

    def _encode_paths(self, paths: torch.Tensor, masks: torch.Tensor, num_paths: torch.Tensor) -> torch.Tensor:
        if len(paths) == 0:
            scores = torch.zeros(1, self.num_relations)
            scores[-1] = 1
            return scores

        # out: batch_size * path_length * hidden_dim
        path_tensors = self.embed_path(paths)
        # out: path_length * batch_size * hidden_dim
        path_tensors = path_tensors.permute(1, 0, 2)
        # out: path_length * batch_size * hidden_dim
        path_embedding = self.transform(path_tensors, src_key_padding_mask=masks)
        # using the CLS token to represent paths
        # out: batch_size * hidden_dims
        path_cls_embedding = path_embedding[0]

        # out: batch_size * num_rels
        path_scores = torch.mm(path_cls_embedding, self.relation_weights.T)
        path_scores = path_scores.sum(dim=1)
        # path_scores = torch.softmax(path_scores, dim=1)

        # summarize multiple paths into one path embedding
        summarized_path_embeddings = []

        relation_scores = []

        for num_path in num_paths:
            assert len(path_cls_embedding) > 0
            path_score_subset = path_scores[:num_path]  # num_paths * num_rels
            path_embeddings = path_cls_embedding[:num_path]  # num_paths * hidden_dim

            if self.training:
                path_weights = torch.sigmoid(path_score_subset).unsqueeze(dim=1)
                weighted_path_embeddings = path_embeddings * path_weights
                summarized_path_embeddings.append(weighted_path_embeddings.mean(dim=0))

                # hackathon formulation
                # relation_embed = torch.mm(path_score_subset.T, triplet_embedding)  # num_rels * hidden_dim
                # relation_score = (relation_embed * self.relation_weights).sum(dim=1)  # num_rels
                # relation_scores.append(relation_score)
            else:
                # find the most "versatile" path
                max_idx = path_score_subset.argmax()
                summarized_path_embeddings.append(path_embeddings[max_idx])

                # hackathon formulation
                # max_idx = path_score_subset.argmax(dim=1)  # num_paths
                # mask = torch.zeros_like(path_score_subset, device=self.device)  # num_paths * num_rels
                # mask[:, max_idx] = 1
                # masked_path_score_subset = mask * path_score_subset  # num_paths * num_rels
                # relation_embed = torch.mm(masked_path_score_subset.T, triplet_embedding)
                # relation_score = (relation_embed * self.relation_weights).sum(dim=1)  # num_rels
                # relation_scores.append(relation_score)

            path_cls_embedding = path_cls_embedding[num_path:]
            path_scores = path_scores[num_path:]

        # out: batch_size * hidden_dim
        return torch.stack(summarized_path_embeddings)

        # return torch.stack(relation_scores)
