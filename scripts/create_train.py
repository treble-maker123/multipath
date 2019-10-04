"""This script creates a training set based on the relations that exist in the dev and test set, then returns a graph
with those relations and their inverses removed.
"""
from multiprocessing import Pool, cpu_count
from random import sample

from utils import read_file_to_data, list_to_triplets

dataset_path = "data/WN18"

dev_path = f"{dataset_path}/dev.txt"
test_path = f"{dataset_path}/test.txt"
graph_path = f"{dataset_path}/graph.txt"

output_graph_path = f"{dataset_path}/new_graph.txt"
output_train_path = f"{dataset_path}/new_train.txt"
pct_per_relations = 0.3  # remove a third for training

if __name__ == "__main__":
    dev_triplets = read_file_to_data(dev_path)
    test_triplets = read_file_to_data(test_path)
    graph_triplets = read_file_to_data(graph_path)
    train_triplets = []
    pool = Pool(max(cpu_count() - 2, 1))

    print(f"There are {len(graph_triplets)} graph triplets before removal.")

    test_relations = list(set(map(lambda x: x[1], dev_triplets + test_triplets)))

    for relation in test_relations:
        graph_relation_triplets = list(filter(lambda x: x[1] == relation, graph_triplets))
        train_triplets += sample(graph_relation_triplets, k=round(len(graph_relation_triplets) * pct_per_relations))

    print(f"Finished building training set of size {len(train_triplets)}, removing training triplets from graph.")

    graph_triplet_set = set(map(tuple, graph_triplets))
    train_triplet_set = set(map(tuple, train_triplets))
    filtered_graph_set = graph_triplet_set - train_triplet_set
    filtered_graph_triplets = list(map(list, filtered_graph_set))

    print(f"There are {len(filtered_graph_triplets)} graph triplets after removal.")

    formatted_train_triplets = list_to_triplets(train_triplets)
    formatted_graph_triplets = list_to_triplets(graph_triplets)

    with open(output_train_path, "w") as file:
        file.writelines(formatted_train_triplets)

    with open(output_graph_path, "w") as file:
        file.writelines(formatted_graph_triplets)
