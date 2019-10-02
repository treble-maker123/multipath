from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from config import configured_device
from lib import Object
from lib.utils import Graph


class Model(Object, nn.Module, ABC):
    def __init__(self):
        Object.__init__(self)
        nn.Module.__init__(self)
        ABC.__init__(self)
        self.device = configured_device

    @abstractmethod
    def forward(self, data: Tensor, graph: Graph) -> Tensor:
        pass

    @abstractmethod
    def loss(self, data: Tensor, labels: Tensor, graph: Graph) -> Tensor:
        pass
