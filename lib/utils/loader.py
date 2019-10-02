from typing import Dict, Tuple, Optional

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from lib import Object
from lib.utils import invalid_collate


class Loader(Object):
    """Builds a PyTorch DataLoader from the given data.
    """

    class CustomDataset(Object, Dataset):
        def __init__(self, data: np.ndarray, label: np.ndarray = None):
            Object.__init__(self)
            Dataset.__init__(self)
            if label is not None:
                assert len(data) == len(label), \
                    f"Size mismatch between data ({len(data)}) and ({len(label)})!"
            self.data = data
            self.label = label

        def __len__(self):
            if self.config and self.config.data_size != -1:
                return self.config.data_size

            return len(self.data)

        def __getitem__(self, idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
            triplets = self.data[idx]
            labels = np.zeros(triplets.shape[0]) if self.label is None else self.label[idx].astype("long")

            # -1 when the entity or relation are not in the vocab file
            if (triplets == -1).sum() > 0:
                return None, None
            else:
                return triplets, labels

    def __init__(self):
        super().__init__()

    @classmethod
    def build(cls, data: np.ndarray, label: np.ndarray = None, **loader_params: Dict) -> DataLoader:
        default_loader_params = {
            "batch_size": 2147483648,
            "shuffle": False,
            "num_workers": 0,
            "collate_fn": invalid_collate
        }
        default_loader_params.update(loader_params)

        dataset = Loader.CustomDataset(data, label)

        return DataLoader(dataset, **default_loader_params)
