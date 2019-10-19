"""This script generates paths for all of the input triplets based on the graph triplets offline. This must be run
from the project root directory because it uses code from lib/.

python scripts/generate_paths.py
"""
import sys

sys.path.append(".")

import torch
from lib.utils.dgl_utils import build_test_graph
from lib.utils import Dataset
from utils import enumerate_paths
from time import time
import ray
import pickle
import os
from pdb import set_trace

dataset_path = "data/nell-995"
input_set = "test"
graph_set = "graph"
output_path = f"{input_set}_paths"
max_hops = 4
limit = -1

# Linux
ray.init(address="localhost:8765")

# MacOS
# num_workers = 4
# worker_memory = int(2e9)  # bytes
# object_memory = int(8e9)  # bytes
# ray.init(num_cpus=num_workers, memory=worker_memory, object_store_memory=object_memory)


@ray.remote
def enumerate_path_wrapper(src_dst_pair, output_file_name, num_hops, pickled_graph, padding_tokens):
    _, path_tensor, mask_tensor = enumerate_paths(src_dst_pair, num_hops, pickled_graph, padding_tokens)

    with open(f"{dataset_path}/{output_file_name}", "wb") as file:
        pickle.dump([path_tensor, mask_tensor], file)


if __name__ == "__main__":
    print(f"Started generating paths with input: {input_set}, graph: {graph_set}, and output: {output_path}.")
    start_time = time()
    max_hops = ray.put(max_hops)
    mapping = Dataset(dataset_path=dataset_path)

    input_triplets = torch.from_numpy(mapping.get(input_set).T)
    graph_triplets = mapping.get(graph_set).T

    graph, relations, _ = \
        build_test_graph(mapping.num_entities, mapping.num_relations, graph_triplets, inverse=False)
    graph.ndata.update({"id": torch.arange(0, mapping.num_entities, dtype=torch.long).view(-1, 1)})
    graph.edata.update({"type": torch.from_numpy(relations)})
    pickled_graph = ray.put(pickle.dumps(graph))

    padding = ray.put([mapping.num_relations, mapping.num_entities])
    src_dst_pairs = list(set(map(lambda x: (x[0].item(), x[2].item()), input_triplets)))
    src_dst_tensors = torch.stack(list(map(torch.LongTensor, src_dst_pairs))).unsqueeze(2)

    if limit != -1:
        src_dst_tensors = src_dst_tensors[:limit]

    # make output directory if it doesn't exist
    if not os.path.exists(f"{dataset_path}/{output_path}"):
        print(f"Creating output directory at {dataset_path}/{output_path}")
        os.mkdir(f"{dataset_path}/{output_path}")

    # because the entity ID changes from run to run, and the entity names are not good as file names, storing a manifest
    # that maps the entity names to the file names
    entity_to_string = mapping.entity_id_to_string_dict
    relation_to_string = mapping.relation_id_to_string_dict
    manifest = {
        "entity_dict": entity_to_string,
        "relation_dict": relation_to_string
    }

    object_ids = []
    print("Initializing object IDs")

    for src_dst_pair in src_dst_tensors:
        ints = (src_dst_pair[0].item(), src_dst_pair[1].item())
        strs = (entity_to_string[ints[0]], entity_to_string[ints[1]])
        file_name = f"{'_'.join(list(map(str, ints)))}.pickle"
        pair_output_path = f"{output_path}/{file_name}"

        object_id = enumerate_path_wrapper.remote(src_dst_pair, pair_output_path, max_hops, pickled_graph, padding)
        # object_id = enumerate_paths(src_dst_pair, 3, graph, [mapping.num_relations, mapping.num_entities])
        object_ids.append(object_id)
        manifest[strs] = pair_output_path

    print("Getting object IDs")
    ray.get(object_ids)

    manifest_file_path = f"{dataset_path}/{output_path}/000manifest.pickle"
    print(f"Writing manifest file to {manifest_file_path}")
    with open(manifest_file_path, "wb") as file:
        pickle.dump(manifest, file)

    end_time = time()
    print(f"Finished generating paths in {round(end_time - start_time)} seconds.")
