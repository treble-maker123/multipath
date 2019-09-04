import numpy as np

from lib import Object, Triplets


class Dataset(Object):
    """Wrapper class around the datasets.
    """

    def __init__(self, dataset_path):
        super().__init__()

        self.triplets = {
            "train": Dataset.load_triplets_from_file(f"{dataset_path}/train.txt"),
            "valid": Dataset.load_triplets_from_file(f"{dataset_path}/dev.txt"),
            "test": Dataset.load_triplets_from_file(f"{dataset_path}/test.txt"),
            "graph": Dataset.load_triplets_from_file(f"{dataset_path}/graph.txt")
        }

        entities = []
        relations = []

        for _, value in self.triplets.items():
            src, rel, dst = value
            entities.extend(src)
            entities.extend(dst)
            relations.extend(rel)

        self.unique_entities = list(set(entities))
        self.unique_relations = list(set(relations))
        self.num_entities = len(self.unique_entities)
        self.num_relations = len(self.unique_relations)

        self.logger.info(f"Constructed {dataset_path} dataset with {self.num_entities} entities and "
                         f"{self.num_relations}.")

        self.entity_vocab, self.relation_vocab = {}, {}

        for idx, entity in enumerate(self.unique_entities):
            self.entity_vocab[entity] = idx
        for idx, relation in enumerate(self.unique_relations):
            self.relation_vocab[relation] = idx

        self.data = {
            "train": self.triplets_to_idx(self.triplets["train"]),
            "valid": self.triplets_to_idx(self.triplets["valid"]),
            "test": self.triplets_to_idx(self.triplets["test"]),
            "graph": self.triplets_to_idx(self.triplets["graph"])
        }

    def get(self, split: str) -> np.ndarray:
        return self.data[split]

    def triplets_to_idx(self, triplets: Triplets) -> np.ndarray:
        """Turns a triplet of src, rel, des into a Numpy array.
        """
        src, rel, dst = triplets
        src_idx = np.array([self.entity_vocab[key] for key in src])
        rel_idx = np.array([self.relation_vocab[key] for key in rel])
        dst_idx = np.array([self.entity_vocab[key] for key in dst])

        return np.vstack([src_idx, rel_idx, dst_idx])

    @classmethod
    def load_triplets_from_file(cls, path: str) -> Triplets:
        """Reads the triplet txt file and return a tuple of three List, each containing either a List of source,
        relation, or destination strings.
        """
        with open(path) as graph_file:
            file_content = graph_file.readlines()
        triplets = list(map(lambda x: x.strip().split('\t'), file_content))
        source, relation, destination = list(zip(*triplets))

        return list(source), list(relation), list(destination)