from abc import ABC, abstractmethod
from typing import Type, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from lib import Object
from lib.models import Model
from lib.utils import Graph, Dataset, Loader, Result


class Engine(Object, ABC):
    def __init__(self):
        Object.__init__(self)
        ABC.__init__(self)

        self.graph: Optional[Graph] = None
        self.model: Optional[Type[Model]] = None
        self.optim: Optional[Adam] = None
        self.device: Optional[torch.device] = None
        self.dataset: Optional[Dataset] = Dataset(self.config.dataset_path)
        self.setup_device()

        model_path = self.config.saved_model_path
        self.model_file_path: str = f"{model_path}/{self.config.run_id}.pt" if model_path != "" else ""

        result_path = self.config.saved_result_path
        self.result_path: str = f"{result_path}/{self.config.run_id}.pt" if result_path != "" else ""

    @abstractmethod
    def provide_train_data(self) -> Tuple[Graph, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def provide_valid_data(self) -> Tuple[Graph, np.ndarray]:
        pass

    @abstractmethod
    def provide_test_data(self) -> Tuple[Graph, np.ndarray]:
        pass

    @abstractmethod
    def setup_model(self, from_path: str = None):
        pass

    def train(self, num_epochs: int, validate: bool = True):
        if self.model is None:
            self.setup_model()
            assert self.model is not None, "Must override parent setup_model() method and supply model to self.model " \
                                           "variable!"
            self.setup_optimizer(self.model)

        best_mrr = float("-inf")

        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}...")

            losses = []
            train_graph, train_data, train_label = self.provide_train_data()
            train_generator = enumerate(self.loop_through_data_for_training(train_data,
                                                                            train_label,
                                                                            train_graph,
                                                                            self.optim,
                                                                            self.model))
            for i, (triplets, labels, loss) in train_generator:
                losses.append(loss.mean().item())

            mean_loss = sum(losses) / len(losses) if len(losses) > 0 else 0
            self.logger.info(f"Training for epoch {epoch + 1} completed, mean loss: {mean_loss}")
            self.tensorboard.add_scalar("train/loss", mean_loss, epoch)

            if validate and epoch % self.config.validate_interval == 0:
                self.logger.info("Performing validation on development set...")
                valid_graph, valid_data = self.provide_valid_data()
                dev_result = self.loop_through_data_for_eval(valid_data, valid_graph, self.model)
                dev_mrr = dev_result.calculate_mrr().item()

                self.logger.info(f"Validation completed for epoch {epoch + 1}, results:")
                self.pretty_print_results(dev_result, "dev", epoch)

                if dev_mrr > best_mrr:
                    self.logger.info(f"Better MRR ({round(dev_mrr, 6)} > {round(best_mrr, 6)})!")
                    best_mrr = dev_mrr
                    if self.config.save_model:
                        self.logger.info("Saving model...")
                        self.save_model_to_file(self.model_file_path)

    def test(self, model_path: str = None):
        self.logger.info(f"Loading model with best MRR...")
        self.setup_model(model_path or self.model_file_path)

        self.logger.info("Starting testing...")
        test_graph, test_data = self.provide_test_data()
        test_result = self.loop_through_data_for_eval(test_data, test_graph, self.model)

        self.logger.info("Testing completed! Results:")
        self.pretty_print_results(test_result, "test")
        if self.config.save_result:
            self.logger.info("Saving results...")
            test_result.save_state(self.result_path)

    def loop_through_data_for_training(self,
                                       data: np.ndarray,
                                       target: np.ndarray,
                                       graph: Graph,
                                       optim: Adam,
                                       model: Type[Model]):
        dataset = Loader.build(data, target)

        for i, (triplets, labels) in enumerate(dataset):
            graph.to(self.device)
            model.to(device=self.device)
            model.train()

            triplets = triplets.long().to(device=self.device)
            labels = labels.float().to(device=self.device)

            loss: Tensor = model.loss(triplets, labels, graph)

            loss.backward()
            clip_grad_norm_(model.parameters(), self.config.grad_norm)
            optim.step()
            optim.zero_grad()

            yield triplets.detach().cpu(), \
                  labels.detach().cpu(), \
                  loss.detach().cpu()

    @torch.no_grad()
    def loop_through_data_for_eval(self,
                                   data: np.ndarray,
                                   graph: Graph,
                                   model: Type[Model]) -> Result:
        dataset = Loader.build(data, batch_size=self.config.test_batch_size)
        graph.to(self.device)
        model.to(device=self.device)
        model.eval()

        result = Result()

        with tqdm(total=len(dataset)) as pbar:
            for i, (triplets, _) in enumerate(dataset):
                triplets = triplets.to(device=self.device)

                scores: Tensor = model.forward(triplets, graph)
                labels = triplets[:, 2]

                result.append(scores.cpu(), labels.cpu())

                pbar.update(1)

        return result

    def setup_optimizer(self, model: Type[Model] = None):
        model = model or self.model

        learn_rate = self.config.learn_rate
        weight_decay = self.config.weight_decay

        params = model.parameters()
        self.optim = Adam(params, lr=learn_rate, weight_decay=weight_decay)

    def save_model_to_file(self, path: str, model: Type[Model] = None):
        if path == "":
            self.logger.info("Model file path is empty, skipping save model.")
            return

        self.logger.info(f"Saving current model to {path}.")
        model = model or self.model
        model = model.cpu()  # for inference on CPU-only machine
        torch.save(model.state_dict(), path)

    def setup_device(self, device: torch.device = None):
        """Setup which device (CPU vs. GPU) to use.
        """
        if device is not None:
            self.logger.info(f"Using {self.device} for training.")
            self.device = device
            return

        num_gpu = torch.cuda.device_count()

        if self.config.use_gpu and num_gpu > 0:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"Using {self.device} for training (GPUs: {num_gpu}).")

    def pretty_print_results(self, result: Result, split: str, epoch: int = 0):
        mrr = result.calculate_mrr().item()
        top_1 = result.calculate_top_hits(hit=1).detach().item()
        top_3 = result.calculate_top_hits(hit=3).detach().item()
        top_10 = result.calculate_top_hits(hit=10).detach().item()

        # self.logger.info(
        print(
            f"{split} results:"
            f"\n\t MRR: {round(mrr, 6)}"
            f"\n\t TOP 1 HIT: {round(top_1, 6)}"
            f"\n\t TOP 3 HIT: {round(top_3, 6)}"
            f"\n\t TOP 10 HIT: {round(top_10, 6)}"
        )

        self.tensorboard.add_scalar(f"{split}/mrr", mrr, epoch)
        self.tensorboard.add_scalars(f"{split}/top_hits", {
            "Top 1": top_1,
            "Top 3": top_3,
            "Top 10": top_10
        }, epoch)