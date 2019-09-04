from typing import Tuple

import numpy as np
import torch

from lib.engines import Engine
from lib.models import RGCN
from lib.utils import Graph
from lib.utils.dgl_utils import build_test_graph, get_adj_and_degrees, generate_sampled_graph_and_labels, perturb_data


class RGCNEngine(Engine):
    def __init__(self):
        super().__init__()

        self.num_nodes = self.dataset.num_entities
        self.num_relations = self.dataset.num_relations
        self.train_data = self.dataset.get("graph").T
        self.valid_data = self.dataset.get("valid").T
        self.test_data = self.dataset.get("test").T

        # setting inverse to false because the graph file already contains inverted edges
        self.test_graph, test_relations, test_norm = \
            build_test_graph(self.num_nodes, self.num_relations, self.train_data, inverse=False)
        self.test_graph.ndata.update({
            "id": torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1),
            "norm": torch.from_numpy(test_norm).view(-1, 1)
        })
        self.test_graph.edata['type'] = torch.from_numpy(test_relations)

        self.adj_list, self.degrees = get_adj_and_degrees(self.num_nodes, self.train_data)

    def provide_train_data(self) -> Tuple[Graph, np.ndarray, np.ndarray]:
        super().provide_train_data()

        graph, node_id, edge_type, node_norm, data, labels = \
            generate_sampled_graph_and_labels(self.train_data,
                                              self.config.graph_sample_size,
                                              self.config.train_graph_split,
                                              self.num_relations,
                                              self.adj_list,
                                              self.degrees,
                                              self.config.negative_sample_factor)

        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        node_norm = torch.from_numpy(node_norm).view(-1, 1)
        graph.ndata.update({"id": node_id, "norm": node_norm})
        graph.edata["type"] = edge_type

        return graph, data, labels

    def provide_valid_data(self) -> Tuple[Graph, np.ndarray]:
        super().provide_valid_data()

        return self.test_graph, perturb_data(self.valid_data.T).T

    def provide_test_data(self) -> Tuple[Graph, np.ndarray]:
        super().provide_test_data()

        return self.test_graph, perturb_data(self.test_data.T).T

    def setup_model(self, from_path: str = None):
        super().setup_model(from_path)

        self.logger.info("Setting up model...")

        self.model = RGCN(self.num_nodes,
                          self.config.hidden_dim,
                          self.num_relations,
                          self.config.num_bases,
                          self.config.num_rgcn_layers,
                          self.config.loop_dropout,
                          self.config.embedding_decay)

        if from_path is not None:
            self.logger.info(f"Loading weights from {from_path}.")
            state_dict = torch.load(from_path)
            self.model.load_state_dict(state_dict)

        self.logger.info("Finished setting up RGCN model.")
