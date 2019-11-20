import pickle
import numpy as np

import torch
from torch import Tensor

from config import configured_device
from lib import Object
from typing import Dict
from pdb import set_trace


class Result(Object):
    """This object takes in a score tensor of shape NUM_SAMPLE X NUM_LABELS and a label of shape NUM_SAMPLE and
    calculates various metrics.
    """

    def __init__(self, entity_dict: Dict = None, relation_dict: Dict = None, scores: Tensor = None,
                 labels: Tensor = None, state: Dict = None):
        super().__init__()

        if scores and labels:
            assert scores.shape[0] == labels.shape[0]

        self.num_gpu = torch.cuda.device_count()
        if state is not None:
            self.state = state
        else:
            self.state = {
                "scores": scores,
                "labels": labels,
                "entity_dict": entity_dict,
                "relation_dict": relation_dict
            }

    def save_state(self, file_path: str):
        if file_path == "":
            self.logger.warning("Result file path is emtpy, skipping saving state.")
            return

        if not self.config.save_result:
            self.logger.warning("Configuration save-result is set to false, skipping saving state.")
            return

        with open(file_path, "wb") as file:
            pickle.dump(self.state, file)

    def append(self, scores: Tensor, labels: Tensor) -> None:
        assert scores.shape[0] == labels.shape[0], f"scores: {scores.shape[0]}, labels: {labels.shape[0]}"

        if self.state["scores"] is None:
            self.state["scores"] = scores
            self.state["labels"] = labels
        else:
            self.state["scores"] = torch.cat([self.state["scores"], scores])
            self.state["labels"] = torch.cat([self.state["labels"], labels])

    def calculate_mrr(self) -> Tensor:
        ranks = self._get_ranks().to(device=configured_device)
        rr: Tensor = 1.0 / ranks.float()
        mrr = rr.mean()

        return mrr.cpu()

    def calculate_top_hits(self, hit=1) -> Tensor:
        ranks = self._get_ranks().to(device=configured_device)
        num_hits = (ranks <= hit).float()
        mean_num_hits = num_hits.mean()

        return mean_num_hits.cpu()

    def _get_ranks(self) -> Tensor:
        score = self.state["scores"].to(device=configured_device)
        label = self.state["labels"].to(device=configured_device)

        _, indices = score.sort(dim=1, descending=True)
        match = indices == label.view(-1, 1)
        ranks = match.nonzero().cpu()

        return ranks[:, 1] + 1  # index from nonzero() starts at 0

    @classmethod
    def load_state(cls, file_path: str) -> 'Result':
        with open(file_path, "rb") as file:
            state = pickle.load(file)

        return cls(state=state)
