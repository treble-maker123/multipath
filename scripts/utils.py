from typing import List, Iterable


def read_file_to_data(file_path: str, split_triplets: bool = False) -> List[List[str]]:
    """Read the triplet file into a list of triplets of the format [src, rel, dst]. Setting split_triplets to true will
    split the list so that the final output is a list of size three, each one containing one src, rel, or dst.
    """
    with open(file_path) as data_file:
        file_content = data_file.readlines()

    triplets = list(map(lambda x: x.strip().split('\t'), file_content))
    return triplets if not split_triplets else list(zip(*triplets))


def list_to_triplets(triplets: List[Iterable[str]]) -> List[str]:
    """Takes a list of triplets in the form of [src, rel, dst] and turn it into a list of string of the format
    "src\trel\tdst\n"
    """
    return list(map(lambda x: "\t".join(x) + "\n", triplets))

