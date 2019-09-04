import numpy as np
from dgl import DGLGraph
from dgl.init import zero_initializer

from lib import Object


class Graph(Object, DGLGraph):
    def __init__(self, num_nodes: int, num_relations: int, triplets: np.ndarray):
        Object.__init__(self)
        DGLGraph.__init__(self)

        assert triplets.shape[0] == 3
        src, rel, dst = triplets

        self.add_nodes(num_nodes)
        self.add_edges(src, dst)

        self.set_n_initializer(zero_initializer)
        self.set_e_initializer(zero_initializer)

        self.logger.info(f"Constructed graph with {num_nodes} nodes and {num_relations} edges.")

    def compute_degree_norm(self) -> np.ndarray:
        num_nodes = self.number_of_nodes()
        in_degree = self.in_degrees(range(num_nodes)).float().numpy()
        norm = 1.0 / in_degree
        norm[np.isinf(norm)] = 0

        return norm
