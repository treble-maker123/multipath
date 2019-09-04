"""Script that provides basic initializations to help with analysis of datasets.
"""
dataset_path = "data/nell-995"
graph_path = f"{dataset_path}/graph.txt"
train_path = f"{dataset_path}/train.txt"
dev_path = f"{dataset_path}/dev.txt"
test_path = f"{dataset_path}/test.txt"


def read_file_to_data(file_path, split_triplets: bool = False):
    with open(file_path) as data_file:
        file_content = data_file.readlines()

    triplets = list(map(lambda x: x.strip().split('\t'), file_content))
    return triplets if not split_triplets else list(zip(*triplets))


if __name__ == "__main__":
    train_triplets = read_file_to_data(train_path)
    dev_triplets = read_file_to_data(dev_path)
    test_triplets = read_file_to_data(test_path)
    graph_triplets = read_file_to_data(graph_path)
