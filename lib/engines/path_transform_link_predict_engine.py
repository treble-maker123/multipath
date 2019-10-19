import pickle

import torch
from torch.utils.data import DataLoader

from lib.engines import Engine
from lib.models import Model, PathTransformLinkPredict
from lib.utils import Graph, Result, PathLoader
from tqdm import tqdm


class PathTransformLinkPredictEngine(Engine):
    def __init__(self):
        super().__init__()
        self.train_graph, _, _ = self.build_graph(self.graph_data)
        self.test_graph, _, _ = self.build_graph(self.graph_data)

        self.entity_id_to_str_dict = self.dataset.entity_id_to_string_dict
        self.relation_id_to_str_dict = self.dataset.relation_id_to_string_dict

        with open(f"{self.config.dataset_path}/train_paths/000manifest.pickle", "rb") as file:
            self.train_manifest = pickle.load(file)
        with open(f"{self.config.dataset_path}/valid_paths/000manifest.pickle", "rb") as file:
            self.valid_manifest = pickle.load(file)
        with open(f"{self.config.dataset_path}/test_paths/000manifest.pickle", "rb") as file:
            self.test_manifest = pickle.load(file)

        self.cls_token = self.dataset.num_entities + 1

        self.entity_mapping = dict((v, k) for k, v in self.train_manifest["entity_dict"].items())
        self.relation_mapping = dict((v, k) for k, v in self.train_manifest["relation_dict"].items())

    def build_model(self, *inputs, **kwargs) -> Model:
        return PathTransformLinkPredict(num_entities=self.num_nodes,
                                        num_relations=self.num_relations,
                                        hidden_dim=self.config.hidden_dim,
                                        num_att_heads=5,
                                        num_transformer_layers=1)

    def run(self) -> None:
        self.train(self.config.num_epochs)
        self.test()

    def train(self, num_epochs: int) -> None:
        train_batch_size = self.config.train_batch_size
        best_mrr = float("-inf")
        train_loader = PathLoader.build(data=self.train_data,
                                        dataset=self.dataset,
                                        manifest=self.train_manifest,
                                        cls_token=self.cls_token,
                                        entity_mapping=self.entity_mapping,
                                        relation_mapping=self.relation_mapping)
        valid_loader = PathLoader.build(data=self.valid_data,
                                        dataset=self.dataset,
                                        manifest=self.valid_manifest,
                                        cls_token=self.cls_token,
                                        entity_mapping=self.entity_mapping,
                                        relation_mapping=self.relation_mapping)

        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}...")

            # ==========================================================================================================
            # Training
            # ==========================================================================================================
            optimizer = self.optimizer
            model = self.model
            model.to(device=self.device)
            model.train()
            epoch_losses = []
            batch_losses = []
            batch_counter = 0

            for idx, (paths, mask, triplet, rel) in enumerate(tqdm(train_loader)):
                if paths.size() == torch.Size([1, 1]):
                    continue

                paths = paths.squeeze(dim=0).to(device=self.device)
                mask = mask.squeeze(dim=0).to(device=self.device)
                triplet = triplet.squeeze(dim=0).to(device=self.device)
                label = rel.squeeze(dim=0).to(device=self.device)

                loss = model.loss(triplet, label, self.train_graph, paths=paths, masks=mask)
                loss.backward()
                batch_losses.append(loss.detach().cpu())
                batch_counter += 1

                if batch_counter == train_batch_size and len(batch_losses) > 0:
                    # self.logger.info("Backpropagating...")
                    optimizer.step()
                    mean_batch_loss = sum(batch_losses) / len(batch_losses)
                    epoch_losses.append(mean_batch_loss)
                    batch_counter = 0

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Training for epoch {epoch + 1} completed, avg loss: {avg_epoch_loss}")
            self.tensorboard.add_scalar("train/loss", avg_epoch_loss, epoch)

            if epoch % self.config.validate_interval == 0:
                if self.config.run_train_during_validate:
                    self.logger.info("Performing validation on training set...")
                    train_result = self.loop_through_data_for_eval(dataset=train_loader,
                                                                   model=self.model,
                                                                   graph=self.test_graph)
                    self.logger.info(f"Validation on training set completed for epoch {epoch + 1}, results: ")
                    self.pretty_print_results(train_result, "dev", epoch)

                self.logger.info("Performing validation on development set...")
                valid_result = self.loop_through_data_for_eval(dataset=valid_loader,
                                                               model=self.model,
                                                               graph=self.test_graph)

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
        valid_loader = PathLoader.build(data=self.test_data,
                                        dataset=self.dataset,
                                        manifest=self.test_manifest,
                                        cls_token=self.cls_token,
                                        entity_mapping=self.entity_mapping,
                                        relation_mapping=self.relation_mapping)

        self.logger.info("Starting testing...")
        test_result = self.loop_through_data_for_eval(dataset=valid_loader,
                                                      model=self.model,
                                                      graph=self.test_graph)

        self.logger.info("Testing completed! Results:")
        self.pretty_print_results(test_result, "test")

        self.logger.info("Saving results...")
        test_result.save_state(self.result_path)

        return test_result

    @torch.no_grad()
    def loop_through_data_for_eval(self,
                                   dataset: DataLoader,
                                   model: Model,
                                   graph: Graph) -> Result:
        graph.to(self.device)
        model.to(device=self.device)
        model.eval()

        result = Result()

        for idx, (paths, mask, triplet, rel) in enumerate(dataset):
            label = rel.squeeze(dim=0).to(device=self.device)

            if paths.size() == torch.Size([1, 1]):
                score = torch.randn(1, self.num_relations + 1)
            else:
                paths = paths.squeeze(dim=0).to(device=self.device)
                mask = mask.squeeze(dim=0).to(device=self.device)
                triplet = triplet.squeeze(dim=0).to(device=self.device)

                score = model(triplet, graph, paths=paths, masks=mask)

            result.append(score.cpu(), label.cpu())

        return result
