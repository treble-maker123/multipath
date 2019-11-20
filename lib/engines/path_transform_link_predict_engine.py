import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.engines import Engine
from lib.models import Model, PathTransformLinkPredict
from lib.utils import Graph, Result, PathLoader
from lib.utils.collate_fn import paths_stack_collate


class PathTransformLinkPredictEngine(Engine):
    def __init__(self):
        super().__init__()
        self.train_graph, _, _ = self.build_graph(self.graph_data)
        self.test_graph, _, _ = self.build_graph(self.graph_data)

        self.entity_id_to_str_dict = self.dataset.entity_id_to_string_dict
        self.relation_id_to_str_dict = self.dataset.relation_id_to_string_dict

        with open(f"{self.config.dataset_path}/train_paths/000manifest.json", "r") as file:
            self.train_manifest = json.load(file)
        with open(f"{self.config.dataset_path}/valid_paths/000manifest.json", "r") as file:
            self.valid_manifest = json.load(file)
        with open(f"{self.config.dataset_path}/test_paths/000manifest.json", "r") as file:
            self.test_manifest = json.load(file)

        self.cls_token = self.dataset.num_entities + 1

    @property
    def train_loader(self):
        self.logger.info("Building training dataset.")
        return PathLoader.build(data=self.train_data,
                                dataset=self.dataset,
                                manifest=self.train_manifest,
                                cls_token=self.cls_token,
                                data_split="train",
                                batch_size=self.config.train_batch_size,
                                num_workers=self.config.num_workers,
                                collate_fn=paths_stack_collate)

    @property
    def valid_train_loader(self):
        self.logger.info("Building training dataset for validation.")
        return PathLoader.build(data=self.train_data,
                                dataset=self.dataset,
                                manifest=self.train_manifest,
                                cls_token=self.cls_token,
                                data_split="valid",
                                batch_size=self.config.test_batch_size,
                                num_workers=self.config.num_workers,
                                collate_fn=paths_stack_collate)

    @property
    def valid_loader(self):
        self.logger.info("Building validation dataset.")
        return PathLoader.build(data=self.valid_data,
                                dataset=self.dataset,
                                manifest=self.valid_manifest,
                                cls_token=self.cls_token,
                                data_split="valid",
                                batch_size=self.config.test_batch_size,
                                num_workers=self.config.num_workers,
                                collate_fn=paths_stack_collate)

    @property
    def test_loader(self):
        self.logger.info("Building test dataset.")
        return PathLoader.build(data=self.test_data,
                                dataset=self.dataset,
                                manifest=self.test_manifest,
                                cls_token=self.cls_token,
                                data_split="test",
                                batch_size=self.config.test_batch_size,
                                num_workers=self.config.num_workers,
                                collate_fn=paths_stack_collate)

    def build_model(self, *inputs, **kwargs) -> Model:
        model = PathTransformLinkPredict(num_entities=self.num_nodes,
                                         num_relations=self.num_relations,
                                         hidden_dim=self.config.hidden_dim,
                                         num_att_heads=self.config.num_attention_heads,
                                         num_transformer_layers=self.config.num_transformer_layers)

        return model

    def run(self) -> None:
        self.train(self.config.num_epochs)
        self.test()

    def train(self, num_epochs: int) -> None:
        best_mrr = float("-inf")
        train_loader = self.train_loader
        valid_train_loader = self.valid_train_loader
        valid_loader = self.valid_loader

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
            batch_loss = 0.0
            path_counter = 0

            for idx, (paths, masks, labels, triplets, num_paths) in enumerate(tqdm(train_loader)):
                bucket_size = self.config.bucket_size
                bucket_generator = self.train_bucket_generator(paths, masks, triplets, labels,
                                                               num_paths=num_paths,
                                                               max_paths_per_bucket=bucket_size)

                for p, m, t, l, n in bucket_generator:
                    assert (p == -1).sum() == 0  # no elements with -1
                    assert p.shape[0] == n.sum().item()  # correct number of paths
                    assert m.shape[0] == n.sum().item()  # correct number of masks

                    p = p.to(device=self.device)
                    m = m.to(device=self.device)
                    t = t.to(device=self.device)
                    l = l.to(device=self.device)
                    n = n.to(device=self.device)

                    loss = model.loss(t, l, self.train_graph, paths=p, masks=m, num_paths=n)

                    loss.backward()
                    batch_loss += loss.detach().cpu()
                    path_counter += len(n)

                optimizer.step()
                batch_loss = batch_loss / path_counter
                epoch_losses.append(batch_loss)

                batch_loss = 0.0
                path_counter = 0

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Training for epoch {epoch + 1} completed, avg loss: {avg_epoch_loss}")
            self.tensorboard.add_scalar("train/loss", avg_epoch_loss, epoch)

            if epoch % self.config.validate_interval == 0:
                if self.config.run_train_during_validate:
                    self.logger.info("Performing validation on training set...")
                    train_result = self.loop_through_data_for_eval(dataset=valid_train_loader,
                                                                   model=self.model,
                                                                   graph=self.test_graph)

                    self.logger.info(f"Validation on training set completed for epoch {epoch + 1}, results: ")
                    self.pretty_print_results(train_result, "train", epoch)

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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)

    def test(self) -> Result:
        self.logger.info(f"Loading model with best MRR...")
        self.model = self.build_model().initialize_weights_from_file(file_path=self.model_file_path)
        test_loader = self.test_loader

        self.logger.info("Starting testing...")
        test_result = self.loop_through_data_for_eval(dataset=test_loader,
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

        result = Result(entity_dict=self.entity_id_to_str_dict, relation_dict=self.relation_id_to_str_dict)

        for idx, (paths, mask, _, triplet, num_paths) in enumerate(tqdm(dataset)):
            labels = triplet[:, 1]
            assert len(triplet) == len(labels)

            if num_paths.size() == torch.Size([1, 1]) and num_paths.item() == 0:
                score = torch.randn(1, self.num_relations)
            else:
                paths = paths.to(device=self.device)
                mask = mask.to(device=self.device)
                triplet = triplet.to(device=self.device)

                score = model(triplet, graph, paths=paths, masks=mask, num_paths=num_paths)

            result.append(score.cpu(), labels.cpu())

        return result

    def train_bucket_generator(self,
                               paths: torch.Tensor,
                               masks: torch.Tensor,
                               triplet: torch.Tensor,
                               labels: torch.Tensor,
                               num_paths: torch.Tensor,
                               max_paths_per_bucket: int = 16000):
        """Divides the input data into buckets with maximum size of max_paths_per_bucket.
        """
        assert len(paths) == len(masks)
        assert len(triplet) == len(labels)

        # paths contain [-1, -1 ...] vector in place of empty paths
        num_empty_paths = (num_paths == 0).nonzero().shape[0]
        assert num_paths.sum().item() == (paths.shape[0] - num_empty_paths)

        bucket = self._create_new_bucket()

        for num_path_tensor in num_paths:
            num_path: int = num_path_tensor.item()

            # skip the ones without paths
            if num_path == 0:
                # they still have a corresponding tensor with -1 values in paths and masks, get rid of those
                paths = paths[1:]
                masks = masks[1:]
                triplet = triplet[1:]
                labels = labels[1:]
                continue

            # if adding this triplet will overflow, yield what we have so far
            if bucket["count"] + num_path > max_paths_per_bucket:
                yield torch.cat(bucket["paths"]), \
                      torch.cat(bucket["masks"]), \
                      torch.stack(bucket["triplet"]), \
                      torch.stack(bucket["labels"]), \
                      torch.stack(bucket["num_paths"])
                bucket = self._create_new_bucket()

            num_paths_before = paths.shape[0]

            # add the current paths to the bucket
            bucket["count"] += num_path
            bucket["paths"].append(paths[:num_path])
            bucket["masks"].append(masks[:num_path])
            bucket["triplet"].append(triplet[0])
            bucket["labels"].append(labels[0])
            bucket["num_paths"].append(num_path_tensor)

            # remove those values from the original tensor
            paths = paths[num_path:]
            masks = masks[num_path:]
            triplet = triplet[1:]
            labels = labels[1:]

            num_paths_after = paths.shape[0]

            assert num_paths_after + num_path == num_paths_before, f"Expecting bucket to remove {num_path}, started" \
                                                                   f"with {num_paths_before}, and ended with " \
                                                                   f"{num_paths_after}"

        assert len(paths) == 0
        assert len(masks) == 0
        assert len(triplet) == 0
        assert len(labels) == 0

        # if there's left over, yield one last time
        if bucket["count"] > 0:
            yield torch.cat(bucket["paths"]), \
                  torch.cat(bucket["masks"]), \
                  torch.stack(bucket["triplet"]), \
                  torch.stack(bucket["labels"]), \
                  torch.stack(bucket["num_paths"])

    @classmethod
    def _create_new_bucket(cls):
        return {
            "count": 0,
            "paths": [],
            "masks": [],
            "triplet": [],
            "labels": [],
            "num_paths": []
        }
