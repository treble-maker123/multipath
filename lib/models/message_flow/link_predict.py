from pdb import set_trace

import dgl.function as fn
from dgl import NodeFlow
from dgl.contrib.sampling.sampler import NeighborSampler
from dgl.udf import EdgeBatch
from torch import Tensor
from torch.nn import functional as F

from lib.models.message_flow.embedding import EntityEmbedding, RelationEmbedding
from lib.models.model import Model
from lib.utils import Graph


class LinkPredict(Model):
    def __init__(self, max_hops: int, num_entities: int, num_relations: int, hidden_dim: int):
        super().__init__()

        self.max_hops = max_hops
        self.embed_entities = EntityEmbedding(num_entities, hidden_dim)
        self.embed_relations = RelationEmbedding(num_relations, hidden_dim)

    def forward(self, data: Tensor, graph: Graph) -> Tensor:
        super().forward(data, graph)

        src_node_id, dst_node_id = data.t()
        path_embedding = self._get_path_embedding(src_node_id, dst_node_id, graph, self.max_hops).unsqueeze(0)
        relation_embeddings = self.embed_relations.embedding.weight

        distance = LinkPredict._pairwise_distance(path_embedding, relation_embeddings)

        return distance

    def loss(self, data: Tensor, labels: Tensor, graph: Graph) -> Tensor:
        super().loss(data, labels, graph)

        # TODO: Mask the label relations?
        # TODO: Consider predicting all of the relationships at the same time?
        # TODO: Maybe select all of the training examples out of the graph first and remove them?
        scores = self.forward(data, graph)

    def _get_path_embedding(self,
                            src_node_id: Tensor,
                            dst_node_id: Tensor,
                            graph: Graph,
                            num_hops: int) -> Tensor:
        subgraph = self.get_path_subgraph(src_node_id, dst_node_id, num_hops, graph)
        subgraph.readonly()
        subgraph.copy_from_parent()

        self.embed_entities(subgraph)
        self.embed_relations(subgraph)

        src_node_idx = (subgraph.ndata["id"].squeeze() == src_node_id).nonzero().squeeze()
        sampler = NeighborSampler(subgraph,
                                  batch_size=1,
                                  num_hops=self.max_hops,
                                  seed_nodes=src_node_idx,
                                  expand_factor=100000,
                                  neighbor_type="out")

        for g in sampler:  # should only be one because there is only one seed node
            node_flow: NodeFlow = g
            node_flow.copy_from_parent()
            # TODO: Is fn.mean the best way? Same as normalizing by in-degree?
            node_flow.prop_flow(message_funcs=self._message_func, reduce_funcs=fn.mean(msg="msg", out="h"))

    @staticmethod
    def _pairwise_distance(path_embedding: Tensor, relations_embedding: Tensor):
        return F.pairwise_distance(path_embedding, relations_embedding, p=2, eps=1e-10)

    @staticmethod
    def _message_func(edges: EdgeBatch):
        set_trace()
        # TODO: *** RuntimeError: index out of range at /Users/distiller/project/conda/conda-bld/pytorch_1556653464916/work/aten/src/TH/generic/THTensorEvenMoreMath.cpp:193
        node_embedding = edges.src["h"]
        edge_embedding = edges.data["h"]
        msg = node_embedding * edge_embedding

        return {"msg": msg}
