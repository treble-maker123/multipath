from abc import ABC, abstractmethod

import torch
from torch import Tensor

from config import configured_device
from lib.models.module import Module
from lib.utils import Graph


class Model(Module, ABC):
    def __init__(self):
        Module.__init__(self)
        ABC.__init__(self)
        self.device = configured_device

    def save_weights_to_file(self, file_path: str) -> None:
        if not self.config.save_model:
            self.logger.warning("Configuration save-model set to false, skipping saving model.")
            return

        self.cpu()
        torch.save(self.state_dict(), file_path)

    def initialize_weights_from_file(self, file_path: str) -> 'Model':
        state_dict = torch.load(file_path)
        self.load_state_dict(state_dict)
        return self

    @abstractmethod
    def forward(self, data: Tensor, graph: Graph, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def loss(self, data: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        pass
