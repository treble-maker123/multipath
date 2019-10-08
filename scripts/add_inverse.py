"""This script adds an inverse triplet to all triplets in the given file by appending INV_TOKEN to the end of the
relation, as well as swapping the position of the two entities in the triplet.
"""
from copy import deepcopy

from utils import read_file_to_data, list_to_triplets

INV_TOKEN = "_inv"
DATASET_PATH = "data/WN18"
FILE_PATH = f"{DATASET_PATH}/graph.txt"

if __name__ == "__main__":
    triplets = read_file_to_data(FILE_PATH)
    triplets_copy = deepcopy(triplets)
    inv_triplets = list(map(lambda x: [x[2], x[1] + INV_TOKEN, x[0]], triplets_copy))

    combined_triplets = list_to_triplets(triplets + inv_triplets)

    with open(f"{DATASET_PATH}/new_graph.txt", "w") as file:
        file.writelines(combined_triplets)
