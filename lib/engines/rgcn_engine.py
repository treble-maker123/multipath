from math import ceil

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from lib.engines import Engine
from lib.models import Model, RGCN
from lib.utils import Graph, Result
from lib.utils.dgl_utils import generate_sampled_graph_and_labels, perturb_data


class RGCNEngine(Engine):
    def __init__(self):
        super().__init__()
        self.test_graph, self.adj_list, self.degrees = self.build_graph(self.graph_data)

    def build_model(self, *inputs, **kwargs) -> Model:
        return RGCN(num_nodes=self.num_nodes,
                    hidden_dim=self.config.hidden_dim,
                    num_relations=self.num_relations,
                    num_bases=self.config.num_bases,
                    dropout=self.config.loop_dropout,
                    num_layers=self.config.num_rgcn_layers,
                    node_regularization_param=self.config.embedding_decay,
                    regularizer=self.config.rgcn_regularizer)

    def run(self) -> None:
        self.train(self.config.num_epochs)
        self.test()

    def train(self, num_epochs: int):
        best_mrr = float("-inf")

        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}...")

            # ==========================================================================================================
            # Training
            # ==========================================================================================================
            self.logger.info("Sampling graph and training data...")
            graph, node_id, edge_type, node_norm, train_data, train_labels = \
                generate_sampled_graph_and_labels(self.graph_data,
                                                  self.config.graph_sample_size,
                                                  self.config.train_graph_split,
                                                  self.num_relations,
                                                  self.adj_list,
                                                  self.degrees,
                                                  self.config.negative_sample_factor)

            node_id = torch.from_numpy(node_id).view(-1, 1).long()
            edge_type = torch.from_numpy(edge_type)
            node_norm = torch.from_numpy(node_norm).view(-1, 1)
            graph.ndata.update({"id": node_id, "norm": node_norm})
            graph.edata["type"] = edge_type

            model = self.model
            optimizer = self.optimizer
            optimizer.zero_grad()

            graph.to(self.device)
            model.to(device=self.device)
            model.train()

            train_data = torch.from_numpy(train_data).long().to(device=self.device)
            train_labels = torch.from_numpy(train_labels).float().to(device=self.device)

            loss = model.loss(train_data, train_labels, graph)
            loss.backward()
            clip_grad_norm_(model.parameters(), self.config.grad_norm)

            optimizer.step()

            loss = loss.detach().cpu().item()

            self.logger.info(f"Training for epoch {epoch + 1} completed, loss: {loss}.")
            self.tensorboard.add_scalar("train/loss", loss, epoch)

            # ==========================================================================================================
            # Validation
            # ==========================================================================================================
            if epoch % self.config.validate_interval == 0:
                self.logger.info("Performing validation on development set...")
                valid_data = perturb_data(self.valid_data.T).T
                valid_result = self.loop_through_data_for_eval(dataset=valid_data,
                                                               model=model,
                                                               graph=self.test_graph,
                                                               batch_size=self.config.test_batch_size)
                valid_mrr = valid_result.calculate_mrr().item()

                self.logger.info(f"Validation completed for epoch {epoch + 1}, results: ")
                self.pretty_print_results(valid_result, "dev", epoch)

                if valid_mrr > best_mrr:
                    self.logger.info(f"Better MRR ({round(valid_mrr, 6)} > {round(best_mrr, 6)})!")
                    best_mrr = valid_mrr
                    self.save_current_model()

    def test(self) -> Result:
        self.logger.info(f"Loading model with best MRR...")
        self.model = self.build_model().initialize_weights_from_file(file_path=self.model_file_path)

        self.logger.info("Starting testing...")
        test_data = perturb_data(self.test_data.T).T
        test_result = self.loop_through_data_for_eval(dataset=test_data,
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
                                   dataset: np.ndarray,  # assuming batch_size * 3
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
            labels = batch[:, 2]  # the objects in <subject, relation, object>

            scores = model(batch, graph)
            result.append(scores.cpu(), labels.cpu())

        return result
