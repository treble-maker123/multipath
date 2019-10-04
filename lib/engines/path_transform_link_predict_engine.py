from typing import Tuple

import numpy as np
import torch

from lib.engines import Engine
from lib.models import PathTransformLinkPredict
from lib.utils import Graph
from lib.utils.dgl_utils import build_test_graph


class PathTransformLinkPredictEngine(Engine):
    def __init__(self):
        super().__init__()

        self.graph, relations, norm = \
            build_test_graph(self.num_nodes, self.num_relations, self.train_data, inverse=False)
        self.graph.ndata.update({
            "id": torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1),
            "norm": torch.from_numpy(norm).view(-1, 1)
        })

        self.graph.edata["type"] = torch.from_numpy(relations)

    def provide_train_data(self) -> Tuple[Graph, np.ndarray, np.ndarray]:
        super().provide_train_data()
        queries = self.train_data[:, [0, 2]]
        answers = self.train_data[:, 1]

        return self.graph, queries, answers

    def provide_valid_data(self) -> Tuple[Graph, np.ndarray, np.ndarray]:
        super().provide_valid_data()
        queries = self.valid_data[:, [0, 2]]
        answers = self.valid_data[:, 1]

        return self.graph, queries, answers

    def provide_test_data(self) -> Tuple[Graph, np.ndarray, np.ndarray]:
        super().provide_test_data()
        queries = self.test_data[:, [0, 2]]
        answers = self.test_data[:, 1]

        return self.graph, queries, answers

    def setup_model(self, from_path: str = None):
        super().setup_model(from_path)

        self.logger.info("Setting up model...")

        self.model = PathTransformLinkPredict(max_hops=self.config.max_traversal_hops,
                                              num_entities=self.num_nodes,
                                              num_relations=self.num_relations,
                                              hidden_dim=self.config.hidden_dim,
                                              num_transformer_layers=3)

        if from_path is not None:
            self.logger.info(f"Loading weights from {from_path}.")
            state_dict = torch.load(from_path)
            self.model.load_state_dict(state_dict)

        self.logger.info("Finished setting up MultiPath link predict model.")
