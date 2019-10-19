import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.object import Object
from lib.utils.dataset import Dataset


class PathLoader(Object, torch.utils.data.Dataset):
    def __init__(self,
                 data: np.ndarray,
                 dataset: Dataset,
                 manifest: Dict[str, object],
                 cls_token: int,
                 entity_mapping: Dict[str, int],
                 relation_mapping: Dict[str, int]):
        Object.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        self.data = torch.from_numpy(data)
        self.dataset = dataset
        self.manifest = manifest
        self.cls_token = cls_token
        # these entity/relation to ID mapping are generated when the paths were generated, therefore they may not
        # match the IDs of the current graph, and conversion is necessary
        self.entity_mapping = entity_mapping
        self.relation_mapping = relation_mapping

        self.entity_id_to_str_dict = self.dataset.entity_id_to_string_dict
        self.relation_id_to_str_dict = self.dataset.relation_id_to_string_dict
        self.default_tensor = torch.LongTensor([-1])

        self.max_paths = self.config.max_paths

    def __len__(self):
        return self.data.shape[0] if self.config.data_size == -1 else self.config.data_size

    def __getitem__(self, idx: int):
        triplet = self.data[idx]
        src, rel, dst = triplet.unsqueeze(1)

        src_dst_pair = (self.entity_id_to_str_dict[src.item()], self.entity_id_to_str_dict[dst.item()])

        # converting to entity and relation IDs generated when the paths were generated
        converted_src = self.entity_mapping[self.entity_id_to_str_dict[src.item()]]
        converted_rel = self.relation_mapping[self.relation_id_to_str_dict[rel.item()]]
        converted_dst = self.entity_mapping[self.entity_id_to_str_dict[dst.item()]]
        converted_triplet = torch.LongTensor([converted_src, converted_rel, converted_dst])

        rel_tensor = torch.LongTensor([converted_rel])

        if src_dst_pair not in self.manifest.keys():
            self.logger.warning(f"{src_dst_pair} not found in train manifest.")
            return self.default_tensor, self.default_tensor, triplet, rel_tensor

        with open(f"{self.config.dataset_path}/{self.manifest[src_dst_pair]}", "rb") as file:
            paths, mask = pickle.load(file)

        # remove the current path from the paths
        filtered_paths, filtered_mask = [], []
        for idx, path in enumerate(paths):
            match = (path[:3] == converted_triplet).sum().item() == 3
            if not match:
                filtered_paths.append(path)
                filtered_mask.append(mask[idx])

        if len(filtered_paths) == 0:
            # self.logger.warning(f"No filtered paths found for {src_dst_pair}.")
            return self.default_tensor, self.default_tensor, triplet, rel_tensor
        else:
            paths = torch.stack(filtered_paths)
            mask = torch.stack(filtered_mask)

        # add a CLS-like token
        cls_tokens = torch.LongTensor([self.cls_token] * paths.shape[0]).unsqueeze(1)
        mask_tokens = torch.BoolTensor([False] * paths.shape[0]).unsqueeze(1)

        paths = torch.cat((cls_tokens, paths), dim=1)
        mask = torch.cat((mask_tokens, mask), dim=1)

        # if paths are greater than max, sample a subset
        num_paths = paths.shape[0]
        if num_paths > self.max_paths:
            # self.logger.info(f"{num_paths} paths, sampling subset of {self.max_paths} paths...")
            subset_idx = torch.randperm(num_paths)[:self.max_paths]
            paths = paths[subset_idx, :]
            mask = mask[subset_idx, :]

        return paths, mask, triplet, rel_tensor

    @classmethod
    def build(cls,
              data: np.ndarray,
              dataset: Dataset,
              manifest: Dict[str, object],
              cls_token: int,
              entity_mapping: Dict[str, int],
              relation_mapping: Dict[str, int],
              **loader_options) -> DataLoader:
        dataset = cls(data, dataset, manifest, cls_token, entity_mapping, relation_mapping)

        default_loader_params = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0
        }

        default_loader_params.update(loader_options)

        return DataLoader(dataset, **default_loader_params)
