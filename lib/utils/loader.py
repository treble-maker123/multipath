from typing import Dict, Tuple, Optional, Callable, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from lib import Object
from lib.types import Path
from lib.utils import invalid_collate


class Loader(Object):
    class CustomDataset(Object, Dataset):
        def __init__(self,
                     data: np.ndarray,
                     label: np.ndarray = None,
                     subgraph_func: Callable = None,
                     subgraph_func_args: Dict[str, object] = None):
            Object.__init__(self)
            Dataset.__init__(self)
            if label is not None:
                assert len(data) == len(label), \
                    f"Size mismatch between data ({len(data)}) and ({len(label)})!"
            self.data = data
            self.label = label
            self.subgraph_func = subgraph_func
            self.subgraph_func_args = subgraph_func_args

        def __len__(self):
            if self.config and self.config.data_size != -1:
                return self.config.data_size

            return len(self.data)

        def __getitem__(self, idx: int) \
                -> Tuple[Optional[Tensor], Optional[Tensor], Optional[List[Path]], Optional[Tensor]]:
            triplets = self.data[idx]
            labels = np.zeros(triplets.shape[0]) if self.label is None else self.label[idx].astype("long")

            if self.subgraph_func is not None and self.subgraph_func_args is not None:
                src_node_id, dst_node_id = torch.LongTensor(triplets).unsqueeze(1)
                self.subgraph_func_args["src_node_id"] = src_node_id
                self.subgraph_func_args["dst_node_id"] = dst_node_id
                subgraph, masks = self.subgraph_func(**self.subgraph_func_args)
                # if there are no subgraphs, subgraphs = [] and masks = torch.Tensor([0]), also model should skip
                if masks is None:
                    masks = torch.zeros(1)
                else:
                    masks = torch.stack(masks)
            else:
                subgraph, masks = torch.zeros(1), torch.zeros(1)

            # -1 when the entity or relation is not in the vocab file
            if (triplets == -1).sum() > 0:
                return None, None, None, None
            else:
                return triplets, labels, subgraph, masks

    @classmethod
    def build(cls,
              data: np.ndarray,
              label: np.ndarray = None,
              subgraph_func: Callable = None,
              subgraph_func_args: Dict[str, object] = None,
              **loader_params: Dict) -> DataLoader:
        default_loader_params = {
            "batch_size": 2147483648,
            "shuffle": False,
            "num_workers": 0,
            "collate_fn": invalid_collate
        }
        default_loader_params.update(loader_params)

        dataset = Loader.CustomDataset(data, label, subgraph_func, subgraph_func_args)

        return DataLoader(dataset, **default_loader_params)
