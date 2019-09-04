import pickle

import torch
from torch import Tensor

from lib import Object


class Result(Object):
    def __init__(self, score: Tensor = None, label: Tensor = None):
        super().__init__()

        if score and label:
            assert score.shape[0] == label.shape[0]

        self.num_gpu = torch.cuda.device_count()
        self.state = {
            "score": score,
            "label": label
        }

    def save_state(self, file_path: str):
        if file_path == "":
            self.logger.info("Result file path is emtpy, skipping save state.")
            return

        with open(file_path, "wb") as file:
            pickle.dump(self.state, file)

    def append(self, score: Tensor, label: Tensor) -> None:
        assert score.shape[0] == label.shape[0], f"scores: {score.shape[0]}, labels: {label.shape[0]}"

        if self.state["score"] is None:
            self.state["score"] = score
            self.state["label"] = label
        else:
            self.state["score"] = torch.cat([self.state["score"], score])
            self.state["label"] = torch.cat([self.state["label"], label])

    def calculate_mrr(self) -> Tensor:
        ranks = self._get_ranks()
        rr: Tensor = 1.0 / ranks.float()
        mrr = rr.mean()

        return mrr

    def calculate_top_hits(self, hit=1) -> Tensor:
        ranks = self._get_ranks()
        num_hits = (ranks <= hit).float()
        mean_num_hits = num_hits.mean()

        return mean_num_hits

    def _get_ranks(self) -> Tensor:
        score = self.state["score"]
        label = self.state["label"]

        _, indices = score.sort(dim=1, descending=True)
        match = indices == label.view(-1, 1)
        ranks = match.nonzero()

        return ranks[:, 1] + 1  # index from nonzero() starts at 0

    @classmethod
    def load_state(cls, file_path: str) -> 'Result':
        with open(file_path, "rb") as file:
            state = pickle.load(file)

        return cls(state)
