import json
from typing import Dict, Union, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.object import Object
from lib.utils.dataset import Dataset


class PathLoader(Object, torch.utils.data.Dataset):
    def __init__(self,
                 data: np.ndarray,
                 dataset: Dataset,
                 manifest: Union[Dict[str, object], Dict[Tuple[str, str], object]],
                 cls_token: int,
                 path_max_length: int = 10,
                 data_split: str = "train"):
        """

        Args:
            data: num_triplets * 3 sized numpy array
            dataset: dataset object constructed by the Engine class
            manifest: a dictionary containing where the path files are
            cls_token: index for the CLS token
            path_max_length: max length of the paths, including the starting CLS token. Default to 10 for 4-hop paths
            data_split: which split of data this path loader is loading, "train", "valid", or "test"
        """
        Object.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        self.dataset = dataset

        if data_split == "train":
            self.data, self.labels = self._negative_sample(data, sample_rate=self.config.negative_sample_factor)
            self.data, self.labels = torch.from_numpy(self.data).long(), torch.from_numpy(self.labels).long()
        else:
            self.data = torch.from_numpy(data).long()
            self.labels = torch.ones(len(data)).long()

        self.manifest = manifest
        self.cls_token = cls_token
        self.path_max_length = path_max_length

        self.entity_id_to_str = self.dataset.entity_id_to_string_dict
        self.relation_id_to_str = self.dataset.relation_id_to_string_dict

        self.default_tensor = torch.LongTensor([[-1] * path_max_length])
        self.data_split = data_split

    def __len__(self):
        return self.data.shape[0] if self.config.data_size == -1 else self.config.data_size

    def __getitem__(self, idx: int):
        triplet = self.data[idx]
        label = self.labels[idx]
        src, rel, dst = triplet.unsqueeze(1)

        src_dst_tuple = self.entity_id_to_str[src.item()], self.entity_id_to_str[dst.item()]
        src_dst_str = ", ".join(src_dst_tuple)

        if src_dst_str not in self.manifest.keys():
            self.logger.warning(f"<{src_dst_str}> not found in train manifest.")
            return self.default_tensor, self.default_tensor.bool(), triplet, label, torch.LongTensor([0])

        with open(f"{self.config.dataset_path}/{self.manifest[src_dst_str]}", "rb") as file:
            str_paths = json.load(file)

        # convert the string paths to int paths
        int_paths = list(map(lambda p: self.dataset.path_to_idx(p), str_paths))

        # remove the target path from the paths
        int_triplet = list(map(lambda x: x.item(), triplet))
        try:
            int_paths.remove(int_triplet)
        except ValueError:
            if not (self.data_split == "train" and label.item() == 0):
                raise AssertionError(f"Failed to remove path for {self.data_split} set with query <{src_dst_str}> with"
                                     f" label {label}")

        # return with random values if no paths left
        if len(int_paths) == 0:
            return self.default_tensor, self.default_tensor.bool(), label, triplet, torch.LongTensor([0])

        # add CLS token to the beginning of each path
        int_paths = list(map(lambda x: [self.cls_token] + x, int_paths))

        # pad the paths to max length
        padded_int_paths = list(map(self._pad_to_max_length, int_paths))

        # Turn them into path and mask tensors
        tensor_conversion = map(lambda x: (torch.LongTensor(x[0]), torch.BoolTensor(x[1])), padded_int_paths)
        paths, masks = list(zip(*tensor_conversion))
        paths, masks = torch.stack(paths), torch.stack(masks)

        if self.data_split == "train" and len(paths) > self.config.max_paths:
            paths, masks = self._sample_subset(paths, masks, self.config.max_paths)
        else:
            paths, masks = self._sample_subset(paths, masks, self.config.bucket_size)  # make sure no OOM on GPU

        return paths, masks, label, triplet, torch.LongTensor([len(paths)])

    def _pad_to_max_length(self, path: List[int]) -> Tuple[List[int], List[int]]:
        assert len(path) <= self.path_max_length, f"Expecting length of path ({len(path)}) to be smaller than " \
                                                  f"{self.path_max_length}."

        rel_pad, ent_pad = self.dataset.num_relations, self.dataset.num_entities
        max_length = self.path_max_length
        mask = [0] * len(path) + [1] * (self.path_max_length - len(path))

        for i in range(max_length):
            if len(path) >= self.path_max_length:
                break

            if i % 2 == 0:  # relation
                path.append(rel_pad)
            else:  # entity
                path.append(ent_pad)

        assert path[-1] == ent_pad or path[-1] == path[-1], f"Expecting last element of path to be the padding " \
                                                            f"token {ent_pad} or the past element {path[-1]} but " \
                                                            f"instead got {path[-1]}."
        assert len(path) <= self.path_max_length
        assert len(mask) <= self.path_max_length
        assert len(path) == len(mask)

        return path, mask

    def _sample_subset(self,
                       paths: torch.Tensor,
                       masks: torch.Tensor,
                       num_to_sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_paths = paths.shape[0]
        subset_idx = torch.randperm(num_paths)[:num_to_sample]

        return paths[subset_idx, :], masks[subset_idx, :]

    def _negative_sample(self, data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.info("Sampling negative examples...")

        if sample_rate == 0:
            return data, np.ones(len(data))

        pos_samples = self.dataset.all_triplets
        neg_samples = np.concatenate([np.copy(data) for _ in range(sample_rate)])
        np.random.shuffle(neg_samples[:, 1])

        valid_neg_samples = list(filter(lambda x: not self._is_member(x, pos_samples), neg_samples))
        valid_neg_samples = np.stack(valid_neg_samples)

        pos_labels = np.ones(data.shape[0])
        neg_labels = np.zeros(valid_neg_samples.shape[0])

        self.logger.info(f"Sampled {len(neg_labels)} negative examples.")

        complete_data = np.concatenate([data, valid_neg_samples])
        complete_labels = np.concatenate([pos_labels, neg_labels])

        assert len(complete_data) == len(complete_labels)

        return complete_data, complete_labels

    @classmethod
    def build(cls,
              data: np.ndarray,
              dataset: Dataset,
              manifest: Dict[str, object],
              cls_token: int,
              data_split: str = "train",
              **loader_options) -> DataLoader:
        dataset = cls(data, dataset, manifest, cls_token, data_split=data_split)

        default_loader_params = {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "collate_fn": None
        }

        default_loader_params.update(loader_options)

        return DataLoader(dataset, **default_loader_params)

    @staticmethod
    def _is_member(subarray: np.ndarray, array: np.ndarray) -> bool:
        """Test whether a triplet is a subarray of array.
        """
        return ((subarray == array).sum(axis=1) == 3).sum() > 0
