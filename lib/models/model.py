from abc import ABC, abstractmethod

import torch.nn as nn

from lib import Object


class Model(Object, nn.Module, ABC):
    def __init__(self):
        Object.__init__(self)
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, triplets, graph):
        assert triplets.shape[1] == 3, f"Triplets must be of the size (BATCH_SIZE, 3), got {triplets.shape} instead."

    @abstractmethod
    def loss(self, triplets, labels, graph):
        assert triplets.shape[1] == 3, f"Triplets must be of the size (BATCH_SIZE, 3), got {triplets.shape} instead."
