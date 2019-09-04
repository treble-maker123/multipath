from torch.nn import Embedding

from lib.models import Module
from lib.utils import Graph


class RGCNEmbedding(Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.embedding = Embedding(num_nodes, hidden_dim)

    def forward(self, graph: Graph) -> None:
        super().forward()

        node_ids = graph.ndata['id'].squeeze()
        node_embedding = self.embedding(node_ids)
        graph.ndata["h"] = node_embedding
