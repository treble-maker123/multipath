from math import ceil
from pdb import set_trace

import numpy as np
import torch

from lib.engines import Engine
from lib.models import Model
from lib.models import PathTransformLinkPredict
from lib.utils import Graph
from lib.utils import Result


class PathTransformLinkPredictEngine(Engine):
    def __init__(self):
        super().__init__()
        self.train_graph, _, _ = self.build_graph(self.graph_data)
        self.test_graph, _, _ = self.build_graph(self.full_graph_data)

    def build_model(self, *inputs, **kwargs) -> Model:
        return PathTransformLinkPredict(max_hops=self.config.max_traversal_hops,
                                        num_entities=self.num_nodes,
                                        num_relations=self.num_relations,
                                        hidden_dim=self.config.hidden_dim,
                                        num_att_heads=5,
                                        num_transformer_layers=6)

    def run(self) -> None:
        self.train(self.config.num_epochs)
        self.test()

    def train(self, num_epochs: int) -> None:
        train_batch_size = self.config.train_batch_size
        num_train_batches = ceil(len(self.train_data) / train_batch_size)
        best_mrr = float("-inf")

        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}...")

            # ==========================================================================================================
            # Training
            # ==========================================================================================================
            optimizer = self.optimizer
            model = self.model
            train_graph = self.train_graph
            train_graph.to(self.device)
            model.to(device=self.device)
            model.train()
            epoch_losses = []

            for batch_idx in range(num_train_batches):
                start_idx, end_idx = batch_idx * train_batch_size, batch_idx * train_batch_size + train_batch_size
                batch = torch.from_numpy(self.train_data[start_idx:end_idx]).long()
                optimizer.zero_grad()

                # TODO: Optimize this later
                losses = []
                for triplet in batch:
                    src, rel, dst = triplet.unsqueeze(1)
                    paths, mask = self.model.enumerate_paths(src_node_id=src,
                                                             dst_node_id=dst,
                                                             max_hops=self.config.max_traversal_hops,
                                                             graph=self.train_graph,
                                                             padding=(self.num_relations, self.num_nodes))

                    if len(paths) == 0:
                        continue

                    paths = torch.LongTensor(paths).to(device=self.device)
                    mask = torch.stack(mask).to(device=self.device)
                    triplet = triplet.long().to(device=self.device)
                    label = rel.long().to(device=self.device)

                    loss = model.loss(triplet, label, train_graph, subgraph=paths, masks=mask)
                    loss.backward()
                    losses.append(loss.detach().cpu())

                optimizer.step()
                avg_loss = sum(losses) / len(losses)
                epoch_losses.append(avg_loss)

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Training for epoch {epoch + 1} completed, avg loss: {avg_epoch_loss}")
            self.tensorboard.add_scalar("train/loss", avg_epoch_loss, epoch)

            if epoch % self.config.validate_interval == 0:
                if self.config.run_train_during_validate:
                    self.logger.info("Performing validation on training set...")
                    train_result = self.loop_through_data_for_eval(dataset=self.train_data,
                                                                   model=self.model,
                                                                   graph=self.test_graph,
                                                                   batch_size=self.config.test_batch_size)
                    self.logger.info(f"Validation on training set completed for epoch {epoch + 1}, results: ")
                    self.pretty_print_results(train_result, "dev", epoch)

                self.logger.info("Performing validation on development set...")
                valid_result = self.loop_through_data_for_eval(dataset=self.valid_data,
                                                               model=self.model,
                                                               graph=self.test_graph,
                                                               batch_size=self.config.test_batch_size)
                valid_mrr = valid_result.calculate_mrr().item()

                self.logger.info(f"Validation on development set completed for epoch {epoch + 1}, results: ")
                self.pretty_print_results(valid_result, "dev", epoch)

                if valid_mrr > best_mrr:
                    self.logger.info(f"Better MRR ({round(valid_mrr, 6)} > {round(best_mrr, 6)})!")
                    best_mrr = valid_mrr
                    self.save_current_model()

    def test(self) -> Result:
        self.logger.info(f"Loading model with best MRR...")
        self.model = self.build_model().initialize_weights_from_file(file_path=self.model_file_path)

        self.logger.info("Starting testing...")
        test_result = self.loop_through_data_for_eval(dataset=self.test_data,
                                                      graph=self.test_graph,
                                                      model=self.model,
                                                      batch_size=self.config.test_batch_size)

        self.logger.info("Testing completed! Results:")
        self.pretty_print_results(test_result, "test")

        self.logger.info("Saving results...")
        test_result.save_state(self.result_path)

        return test_result

    @torch.no_grad()
    def loop_through_data_for_eval(self,
                                   dataset: np.ndarray,
                                   model: Model,
                                   graph: Graph,
                                   batch_size: int) -> Result:
        graph.to(self.device)
        model.to(device=self.device)
        model.eval()

        result = Result()

        num_batches = ceil(batch_size / len(dataset))

        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * batch_size, batch_idx * batch_size + batch_size
            batch = torch.from_numpy(dataset[start_idx:end_idx]).long().to(device=self.device)

            for triplet in batch:
                src, rel, dst = triplet.unsqueeze(1)
                paths, mask = self.model.enumerate_paths(src_node_id=src,
                                                         dst_node_id=dst,
                                                         max_hops=self.config.max_traversal_hops,
                                                         graph=self.train_graph,
                                                         padding=(self.num_relations, self.num_nodes))
                paths = torch.LongTensor(paths).to(device=self.device)
                set_trace()
                mask = torch.stack(mask).to(device=self.device)
                triplet = triplet.long().to(device=self.device)
                label = rel.long().to(device=self.device)

                if len(paths) == 0:
                    score = torch.randn(1, self.num_relations + 1)
                else:
                    score = model(triplet, graph, subgraph=paths, masks=mask)

                result.append(score.cpu(), label.cpu())

        return result
