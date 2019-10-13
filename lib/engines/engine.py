from abc import ABC, abstractmethod
from typing import Optional, Iterable, Dict, Tuple, List

import numpy as np
import torch
import torch.optim as optim

from config import configured_device
from lib import Object
from lib.models import Model
from lib.types import Optimizer
from lib.utils import Dataset, Result
from lib.utils import Graph
from lib.utils.dgl_utils import build_test_graph, get_adj_and_degrees


class Engine(Object, ABC):
    def __init__(self):
        Object.__init__(self)
        ABC.__init__(self)

        self.dataset: Optional[Dataset] = Dataset(self.config.dataset_path)
        self.model: Optional[Model] = self.build_model()
        self.optimizer: Optional[Optimizer] = Engine.build_optimizer(**self.build_optimizer_params())

    # ==================================================================================================================
    # Properties
    # ==================================================================================================================

    @property
    def num_gpus(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return configured_device

    @property
    def model_file_path(self) -> str:
        model_path = self.config.saved_model_path
        return f"{model_path}/{self.config.run_id}.pt" if model_path != "" else ""

    @property
    def result_path(self) -> str:
        result_path = self.config.saved_result_path
        return f"{result_path}/{self.config.run_id}.pt" if result_path != "" else ""

    @property
    def train_data(self) -> np.ndarray:
        data = self.dataset.get("train").T
        return data if self.config.data_size == -1 else data[:self.config.data_size]

    @property
    def valid_data(self) -> np.ndarray:
        data = self.dataset.get("valid").T
        return data if self.config.data_size == -1 else data[:self.config.data_size]

    @property
    def test_data(self) -> np.ndarray:
        data = self.dataset.get("test").T
        return data if self.config.data_size == -1 else data[:self.config.data_size]

    @property
    def graph_data(self) -> np.ndarray:
        return self.dataset.get("graph").T

    @property
    def full_graph_data(self) -> np.ndarray:
        return self.dataset.get("full_graph").T

    @property
    def num_nodes(self) -> int:
        return self.dataset.num_entities

    @property
    def num_relations(self) -> int:
        return self.dataset.num_relations

    def build_graph(self, graph_data: np.ndarray) -> Tuple[Graph, List[np.ndarray], np.ndarray]:
        graph, relations, norm = \
            build_test_graph(self.num_nodes, self.num_relations, graph_data, inverse=False)
        graph.ndata.update({
            "id": torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1),
            "norm": torch.from_numpy(norm).view(-1, 1)
        })
        graph.edata['type'] = torch.from_numpy(relations)
        adj_list, degrees = get_adj_and_degrees(self.num_nodes, graph_data)

        return graph, adj_list, degrees

    def save_current_model(self) -> None:
        if not self.config.save_model:
            self.logger.info("Configuration save-model set to false, skipping checkpoint.")
            return
        self.model.save_weights_to_file(self.model_file_path)

    def pretty_print_results(self, result: Result, split: str, epoch: int = 0) -> None:
        mrr = result.calculate_mrr().item()
        top_1 = result.calculate_top_hits(hit=1).detach().item()
        top_3 = result.calculate_top_hits(hit=3).detach().item()
        top_10 = result.calculate_top_hits(hit=10).detach().item()

        self.logger.info(
            f"Epoch {epoch} {split} results:"
            f"\n\t MRR: {round(mrr, 6)}"
            f"\n\t TOP 1 HIT: {round(top_1, 6)}"
            f"\n\t TOP 3 HIT: {round(top_3, 6)}"
            f"\n\t TOP 10 HIT: {round(top_10, 6)}"
        )

        self.tensorboard.add_scalar(f"{split}/mrr", mrr, epoch)
        self.tensorboard.add_scalars(f"{split}/top_hits", {
            "Top 1": top_1,
            "Top 3": top_3,
            "Top 10": top_10
        }, epoch)

    # ==================================================================================================================
    # Abstract methods
    # ==================================================================================================================

    @abstractmethod
    def build_model(self, *inputs, **kwargs) -> Model:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    # ==================================================================================================================
    # Optimizer methods
    # ==================================================================================================================

    OPTIMIZER_TYPES = {
        "adam": optim.Adam
    }

    def build_optimizer_params(self, **updated_args) -> Dict[str, object]:
        default_args = {
            "parameters": self.model.parameters(),
            "optimizer_type": "adam",
            "lr": self.config.learn_rate,
            "weight_decay": self.config.weight_decay,
            "momentum": 0.0  # not needed for now
        }

        default_args.update(updated_args)

        return default_args

    @classmethod
    def build_optimizer(cls, parameters: Iterable[object], optimizer_type: str, **optimizer_params) -> Optimizer:
        optimizer_type = optimizer_type.lower()

        if optimizer_type == "adam":
            del optimizer_params["momentum"]

        return Engine.OPTIMIZER_TYPES[optimizer_type](parameters, **optimizer_params)
