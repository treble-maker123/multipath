"""This script removes the development triplets from the NELL-995 dataset from MINERVA repository.
"""

if __name__ == "__main__":
    dataset_path = "data/nell-995"

    with open(f"{dataset_path}/graph.txt") as file:
        file_content = file.readlines()

    graph_triplets = list(map(lambda x: x.strip().split('\t'), file_content))
    graph_triplets = list(map(lambda x: " ".join(x), graph_triplets))
    print(f"There are {len(graph_triplets)} triplets in the graph.")

    with open(f"{dataset_path}/dev.txt") as file:
        file_content = file.readlines()

    dev_triplets = list(map(lambda x: x.strip().split('\t'), file_content))
    dev_triplets = list(map(lambda x: " ".join(x), dev_triplets))
    print(f"There are {len(dev_triplets)} triplets in the test set.")

    for dev_triplet in dev_triplets:
        graph_triplets.remove(dev_triplet)
    print(f"There are {len(graph_triplets)} left in the graph.")

    graph_triplets = list(map(lambda x: "\t".join(x.split(" ")) + "\n", graph_triplets))

    with open(f"{dataset_path}/graph_without_dev.txt", "w") as file:
        file.writelines(graph_triplets)
