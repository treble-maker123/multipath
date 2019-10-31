import pickle
from typing import Dict, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.object import Object
from lib.utils.dataset import Dataset
from scripts.utils import pad_to_max_length
from pdb import set_trace


class PathLoader(Object, torch.utils.data.Dataset):
    def __init__(self,
                 data: np.ndarray,
                 dataset: Dataset,
                 manifest: Union[Dict[str, object], Dict[Tuple[str, str], object]],
                 cls_token: int,
                 path_max_length: int = 10,
                 test_set: bool = False):
        """

        Args:
            data: num_triplets * 3 sized numpy array
            dataset: dataset object constructed by the Engine class
            manifest: a dictionary containing where the path files are
            cls_token: index for the CLS token
            entity_mapping:
            relation_mapping:
            path_max_length: max length of the paths, including the starting CLS token. Default to 10 for 4-hop paths.
        """
        Object.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        self.data = torch.from_numpy(data)
        self.dataset = dataset
        self.manifest = manifest
        self.cls_token = cls_token
        self.path_max_length = path_max_length
        # these entity/relation to ID mapping are generated when the paths were generated, therefore they may not
        # match the IDs of the current graph, and conversion is necessary
        self.entity_mapping = dict((v, k) for k, v in self.manifest["entity_dict"].items())
        self.relation_mapping = dict((v, k) for k, v in self.manifest["relation_dict"].items())
        # these entity/relation to ID mapping are generated for each run
        self.entity_id_to_str_dict = self.dataset.entity_id_to_string_dict
        self.relation_id_to_str_dict = self.dataset.relation_id_to_string_dict

        self.default_tensor = torch.LongTensor([[-1] * path_max_length])
        self.test_set = test_set

    def __len__(self):
        return self.data.shape[0] if self.config.data_size == -1 else self.config.data_size

    def __getitem__(self, idx: int):
        triplet = self.data[idx]
        src, rel, dst = triplet.unsqueeze(1)

        src_dst_pair = self.entity_id_to_str_dict[src.item()], self.entity_id_to_str_dict[dst.item()]

        # converting to the entity and relation IDs used when the paths were generated
        converted_src = self.entity_mapping[self.entity_id_to_str_dict[src.item()]]
        converted_rel = self.relation_mapping[self.relation_id_to_str_dict[rel.item()]]
        converted_dst = self.entity_mapping[self.entity_id_to_str_dict[dst.item()]]
        converted_triplet = torch.LongTensor([converted_src, converted_rel, converted_dst])

        num_paths = 0
        rel_tensor = torch.LongTensor([converted_rel])

        if src_dst_pair not in self.manifest.keys():
            self.logger.warning(f"{src_dst_pair} not found in train manifest.")
            return self.default_tensor, self.default_tensor.bool(), \
                   converted_triplet, rel_tensor, torch.LongTensor([num_paths])

        with open(f"{self.config.dataset_path}/{self.manifest[src_dst_pair]}", "rb") as file:
            paths, masks = pickle.load(file)

        # remove the current path from the paths
        filtered_paths, filtered_mask = [], []
        for idx, path in enumerate(paths):
            match = (path[:3] == converted_triplet).sum().item() == 3
            if not match:
                filtered_paths.append(path)
                filtered_mask.append(masks[idx])

        if len(filtered_paths) == 0:
            # self.logger.warning(f"No filtered paths found for {src_dst_pair}.")
            return self.default_tensor, self.default_tensor.bool(), converted_triplet, \
                   rel_tensor, torch.LongTensor([num_paths])

        # paths contain ALL paths, code above should remove one path that is the target of a query to mask it
        if not self.test_set:  # test set do not have target path, therefore we don't need to remove it
            assert len(filtered_paths) + 1 == len(paths)

        paths = torch.stack(filtered_paths)
        masks = torch.stack(filtered_mask)

        num_paths = paths.shape[0]

        # add a CLS token
        cls_tokens, mask_tokens = [self.cls_token] * num_paths, [False] * num_paths
        cls_tokens = torch.LongTensor(cls_tokens).unsqueeze(1)
        mask_tokens = torch.BoolTensor(mask_tokens).unsqueeze(1)

        paths = torch.cat((cls_tokens, paths), dim=1)
        masks = torch.cat((mask_tokens, masks), dim=1)

        path_length = paths.shape[1]

        # if the path length is less than max, pad the paths to max length
        if path_length < self.path_max_length:
            rel_pad, ent_pad = self.dataset.num_relations, self.dataset.num_entities
            padded_paths = []
            padded_masks = []

            for path in paths:
                # convert tensor to tuple of ints
                path = tuple(map(lambda x: x.item(), list(path.squeeze())))

                padded_path, padded_mask = pad_to_max_length(path,
                                                             max_length=self.path_max_length,
                                                             padding=(rel_pad, ent_pad))

                padded_paths.append(torch.LongTensor(padded_path))
                padded_masks.append(padded_mask)

            paths = torch.stack(padded_paths)
            masks = torch.stack(padded_masks)

        # if paths are greater than max, sample a subset
        if num_paths > self.config.max_paths:
            # self.logger.info(f"{num_paths} paths, sampling subset of {self.config.max_paths} paths...")
            subset_idx = torch.randperm(num_paths)[:self.config.max_paths]
            paths = paths[subset_idx, :]
            masks = masks[subset_idx, :]

        # # always take the first n paths
        # paths = paths[:self.config.max_paths, :]
        # masks = masks[:self.config.max_paths, :]

        num_paths = paths.shape[0]

        return paths, masks, converted_triplet, rel_tensor, torch.LongTensor([num_paths])

    @classmethod
    def build(cls,
              data: np.ndarray,
              dataset: Dataset,
              manifest: Dict[str, object],
              cls_token: int,
              test_set: bool = False,
              **loader_options) -> DataLoader:
        dataset = cls(data, dataset, manifest, cls_token, test_set=test_set)

        default_loader_params = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "collate_fn": None
        }

        default_loader_params.update(loader_options)

        return DataLoader(dataset, **default_loader_params)
