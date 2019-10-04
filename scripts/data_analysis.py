"""Script that provides basic initializations to help with analysis of datasets.
"""
from utils import read_file_to_data

dataset_path = "data/FB15K-237"
graph_path = f"{dataset_path}/graph.txt"
train_path = f"{dataset_path}/train.txt"
dev_path = f"{dataset_path}/dev.txt"
test_path = f"{dataset_path}/test.txt"


if __name__ == "__main__":
    train_triplets = read_file_to_data(train_path)
    dev_triplets = read_file_to_data(dev_path)
    test_triplets = read_file_to_data(test_path)
    graph_triplets = read_file_to_data(graph_path)
