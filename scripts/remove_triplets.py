"""This script removes certain subsets of triplets from the graph.
"""
from utils import read_file_to_data, list_to_triplets


if __name__ == "__main__":
    dataset_path = "data/WN18RR"

    graph_triplets = read_file_to_data(f"{dataset_path}/graph.txt")
    graph_triplet_set = set(list(map(tuple, graph_triplets)))
    print(f"There are {len(graph_triplets)} triplets in the graph set.")

    input_triplets = read_file_to_data(f"{dataset_path}/train.txt")
    input_triplet_set = set(list(map(tuple, input_triplets)))
    print(f"There are {len(input_triplets)} triplets in the triplets set.")

    filtered_graph_triplets = list(graph_triplet_set - input_triplet_set)
    print(f"There are {len(filtered_graph_triplets)} left in the graph.")

    filtered_graph_triplets = list_to_triplets(filtered_graph_triplets)

    with open(f"{dataset_path}/new_graph.txt", "w") as file:
        file.writelines(filtered_graph_triplets)
