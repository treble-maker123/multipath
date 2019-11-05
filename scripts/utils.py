import pickle
from itertools import chain
from typing import List, Iterable
from typing import Tuple, Dict

import torch

enumerate_path_correct_size = torch.Size([2, 1])


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


def enumerate_paths(src_dst_pair: torch.Tensor, max_hops: int, graph,
                    entity_id_to_string_dict: Dict[int, str],
                    relation_id_to_string_dict: Dict[int, str]) -> List[List[str]]:
    """Enumerates all paths composed of edges start on src_node_id, and expanding breadth-first toward dst_node_id,
    with a maximum of max_hops.

    Args:
        src_dst_pair: A 2x1 torch tensor where the first element contains the source node ID and the second element
            contains the destination node ID
        max_hops: The most number of hops to make to identify paths
        graph: The graph on which to traverses
    """
    assert src_dst_pair.size() == enumerate_path_correct_size

    if type(graph).__name__ != "Graph":
        graph = pickle.loads(graph)

    # make sure they are all in the same place
    src_node_id, dst_node_id = src_dst_pair[0].cpu(), src_dst_pair[1].cpu()

    # setting the starting place
    forward_hops = [[(src_node_id.item(),)]]
    backward_hops = [[(dst_node_id.item(),)]]

    # traversing from the source and destination at the same time
    forward_frontier_nodes = src_node_id
    backward_frontier_nodes = dst_node_id

    for nth_hop in range(max_hops):
        # 0, 2, etc. going from source, 1, 3, etc. going from destination
        forward = True if nth_hop % 2 == 0 else False
        current_frontier_nodes = forward_frontier_nodes if forward else backward_frontier_nodes
        current_hop = forward_hops if forward else backward_hops

        # filter out the frontier nodes that are also src_node or dst_node so that we aren't appending to them anymore
        filter_nodes = dst_node_id if forward else src_node_id
        current_frontier_nodes = current_frontier_nodes[(current_frontier_nodes != filter_nodes).nonzero()].squeeze()

        # expand from frontier nodes
        if forward:
            hop_src_ids, hop_dst_ids, hop_edge_ids = graph.out_edges(current_frontier_nodes, form="all")
        else:
            hop_dst_ids, hop_src_ids, hop_edge_ids = graph.in_edges(current_frontier_nodes, form="all")

        # convert the edge indices to relation IDs, node IDs and entity IDs are the same
        rel_ids = graph.edata["type"][hop_edge_ids]
        paths_from_previous_hop = current_hop[-1]

        # create paths with the nodes currently hopped to
        arguments = [hop_src_ids, hop_dst_ids, rel_ids, [paths_from_previous_hop] * len(rel_ids)]
        # chain to get rid of nested lists, set to get rid of duplicates
        hop_paths = list(set(chain(*map(create_hop_paths, *arguments))))

        if forward:
            forward_hops.append(hop_paths)
            forward_frontier_nodes = hop_dst_ids
        else:
            backward_hops.append(hop_paths)
            backward_frontier_nodes = hop_dst_ids

    forward_paths = forward_hops[-1]
    backward_paths = backward_hops[-1]

    # find the intersection between the end of the forward paths and the start of the backward paths
    forward_end_nodes = set(map(lambda x: x[-1], forward_paths))
    backward_end_nodes = set(map(lambda x: x[-1], backward_paths))  # x[-1] as well because paths starting with dst node
    intersection = forward_end_nodes.intersection(backward_end_nodes)

    # filter out all forward and backward paths that do not end in the intersection
    filtered_forward_paths = list(filter(lambda x: x[-1] in intersection, forward_paths))
    filtered_backward_paths = list(filter(lambda x: x[-1] in intersection, backward_paths))

    # flip the direction of the backward paths
    flipped_backward_paths = list(map(lambda x: tuple(reversed(x)), filtered_backward_paths))

    # connect the forward and backward paths
    arguments = [filtered_forward_paths, [flipped_backward_paths] * len(filtered_forward_paths)]
    candidate_paths = list(set(chain(*map(connect_forward_backward_paths, *arguments))))

    # include the paths that are less than max hops away
    shorter_paths = list(filter(lambda path: path[-1] == dst_node_id, list(chain(*forward_hops[1:]))))
    candidate_paths += shorter_paths
    final_paths = []

    for int_path in candidate_paths:
        str_path = []
        for idx, element in enumerate(int_path):
            if idx % 2 == 0:  # entity
                str_path.append(entity_id_to_string_dict[int_path[idx]])
            else:  # relation
                str_path.append(relation_id_to_string_dict[int_path[idx]])
        final_paths.append(str_path)

    print(f"found {len(candidate_paths)} paths.")

    return final_paths


def create_hop_paths(hop_src_id: torch.Tensor,
                     hop_dst_id: torch.Tensor,
                     hop_edge_id: torch.Tensor,
                     paths_from_previous_hop):
    """Extend the paths from previous hop with the new (hop_src_id, hop_dst_id, hop_edge_id) path.
    """
    hop_edge_id, hop_dst_id = hop_edge_id.item(), hop_dst_id.item()
    return [
        (*previous_path, hop_edge_id, hop_dst_id)
        for previous_path in paths_from_previous_hop
        if previous_path[-1] == hop_src_id
    ]


def connect_forward_backward_paths(forward_path,
                                   backward_paths):
    """Connect the forward path with backward paths on the condition that the start of the fowrard_path is the end of
    the backward_path.
    """
    return [
        (*forward_path, *backward_path[1:])
        for backward_path in backward_paths
        if forward_path[-1] == backward_path[0]
    ]


def pad_to_max_length(path: Tuple[int],
                      max_length: int,
                      padding: Tuple[int, int]) -> Tuple[Tuple[int], torch.Tensor]:
    """Pad the given path with the padding tokens (relation, entity), such that they're at max length.
    """
    num_to_pad = (max_length - len(path))
    # divide by 2 because extension consists of 2 elements
    padded_path = path + tuple(padding * (num_to_pad // 2))
    mask = torch.zeros(max_length, dtype=torch.bool)

    if num_to_pad > 0:
        mask[-num_to_pad:] = 1

    return padded_path, mask
