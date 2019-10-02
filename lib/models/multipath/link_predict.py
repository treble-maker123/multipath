from pdb import set_trace
from typing import Union

import dgl.function as fn
from dgl.subgraph import DGLSubGraph
from dgl.udf import EdgeBatch
from torch import Tensor
from torch.nn import functional as F

from lib.models.multipath.embedding import EntityEmbedding, RelationEmbedding
from lib.models.multipath.multi_path_base_model import MultiPathBaseModel
from lib.utils import Graph


class LinkPredict(MultiPathBaseModel):
    def __init__(self, max_hops: int, num_entities: int, num_relations: int, hidden_dim: int):
        super().__init__(max_hops)

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
        path_subgraph = self.get_path_subgraph(src_node_id, dst_node_id, num_hops, graph)
        path_subgraph.copy_from_parent()

        self.embed_entities(path_subgraph)
        self.embed_relations(path_subgraph)

        # TODO: May not be the best way to approach this, the dst_node will see neighbor messages multiple times.
        #  Maybe look into implementing NodeFlow
        for nth_hop in range(self.max_hops):
            LinkPredict._propagate(path_subgraph)

        dst_node_idx = (path_subgraph.ndata['id'].squeeze() == dst_node_id).nonzero().squeeze()

        return path_subgraph.ndata['h'][dst_node_idx]

    @staticmethod
    def _pairwise_distance(path_embedding: Tensor, relations_embedding: Tensor):
        return F.pairwise_distance(path_embedding, relations_embedding, p=2, eps=1e-10)

    @staticmethod
    def _message_func(edges: EdgeBatch):
        node_embedding = edges.src["h"]
        edge_embedding = edges.data["h"]
        msg = node_embedding * edge_embedding

        return {"msg": msg}

    @staticmethod
    def _propagate(graph: Union[Graph, DGLSubGraph]):
        graph.update_all(LinkPredict._message_func, fn.mean(msg='msg', out='h'))
