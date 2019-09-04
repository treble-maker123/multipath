from abc import ABC, abstractmethod

import torch.nn as nn

from lib import Object


class Module(Object, nn.Module, ABC):
    def __init__(self):
        Object.__init__(self)
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
