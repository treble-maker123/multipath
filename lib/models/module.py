from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from lib import Object


class Module(Object, nn.Module, ABC):
    def __init__(self):
        Object.__init__(self)
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @classmethod
    def _get_entity_idx(cls, length: int) -> Tensor:
        """Returns the location of the entities in a path Tensor
        """
        return torch.arange((length + 1) // 2) * 2

    @classmethod
    def _get_relation_idx(cls, length: int) -> Tensor:
        """Returns the location of the relations in a path Tensor
        """
        return torch.arange(length // 2) * 2 + 1
