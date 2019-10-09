import torch

from lib.engines import Engine
from lib.utils.dgl_utils import build_test_graph


class PathTransformLinkPredictEngine(Engine):
    def __init__(self):
        super().__init__()

        self.graph, relations, norm = \
            build_test_graph(self.num_nodes, self.num_relations, self.graph_data, inverse=False)

        self.graph.ndata.update({
            "id": torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1),
            "norm": torch.from_numpy(norm).view(-1, 1)
        })

        self.graph.edata["type"] = torch.from_numpy(relations)

    # @property
    # def subgraph_func_args(self) -> Dict[str, object]:
    #     return {
    #         "max_hops": self.config.max_traversal_hops,
    #         "graph": self.graph,
    #         "padding": (self.num_relations, self.num_nodes)
    #     }
    #
    # def provide_train_data(self) -> Tuple[Graph, DataLoader]:
    #     super().provide_train_data()
    #     queries = self.train_data[:, [0, 2]]
    #     answers = self.train_data[:, 1]
    #
    #     dataset = Loader.build(queries, answers,
    #                            batch_size=self.config.train_batch_size,
    #                            subgraph_func=Model.enumerate_paths,
    #                            subgraph_func_args=self.subgraph_func_args,
    #                            num_workers=self.config.num_workers)
    #     return self.graph, dataset
    #
    # def provide_valid_data(self) -> Tuple[Graph, DataLoader]:
    #     super().provide_valid_data()
    #     queries = self.valid_data[:, [0, 2]]
    #     answers = self.valid_data[:, 1]
    #
    #     dataset = Loader.build(queries, answers,
    #                            batch_size=self.config.test_batch_size,
    #                            subgraph_func=Model.enumerate_paths,
    #                            subgraph_func_args=self.subgraph_func_args,
    #                            num_workers=self.config.num_workers)
    #     return self.graph, dataset
    #
    # def provide_test_data(self) -> Tuple[Graph, DataLoader]:
    #     super().provide_test_data()
    #     queries = self.test_data[:, [0, 2]]
    #     answers = self.test_data[:, 1]
    #
    #     # using the graph including training set to evaluate test
    #     test_graph, relations, norm = \
    #         build_test_graph(self.num_nodes, self.num_relations, self.dataset.get("old_graph").T, inverse=False)
    #
    #     test_graph.ndata.update({
    #         "id": torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1),
    #         "norm": torch.from_numpy(norm).view(-1, 1)
    #     })
    #
    #     test_graph.edata["type"] = torch.from_numpy(relations)
    #
    #     dataset = Loader.build(queries, answers,
    #                            batch_size=self.config.test_batch_size,
    #                            subgraph_func=Model.enumerate_paths,
    #                            subgraph_func_args=self.subgraph_func_args,
    #                            num_workers=self.config.num_workers)
    #     return test_graph, dataset
    #
    # def setup_model(self, from_path: str = None):
    #     super().setup_model(from_path)
    #
    #     self.logger.info("Setting up model...")
    #
    #     self.model = PathTransformLinkPredict(max_hops=self.config.max_traversal_hops,
    #                                           num_entities=self.num_nodes,
    #                                           num_relations=self.num_relations,
    #                                           hidden_dim=self.config.hidden_dim,
    #                                           num_att_heads=5,
    #                                           num_transformer_layers=3)
    #
    #     if from_path is not None:
    #         self.logger.info(f"Loading weights from {from_path}.")
    #         state_dict = torch.load(from_path)
    #         self.model.load_state_dict(state_dict)
    #
    #     self.logger.info("Finished setting up PathTransform link predict model.")
