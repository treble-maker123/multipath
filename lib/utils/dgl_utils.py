"""Methods from the DGL implementation of RGCN.
"""
from typing import Tuple, List

import numpy as np

from lib.utils import Graph


def perturb_data(triplets: np.ndarray) -> np.ndarray:
    src, rel, dst = triplets
    perturb_subject = np.stack((dst, rel, src))
    perturb_object = np.stack((src, rel, dst))

    return np.concatenate((perturb_object, perturb_subject), axis=1)


def build_graph_from_triplets(num_nodes: int,
                              num_rels: int,
                              triplets: np.ndarray,
                              inverse: bool = False) -> Tuple[Graph, np.ndarray, np.ndarray]:
    src, rel, dst = triplets

    if inverse:
        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + num_rels))
        edges = sorted(zip(dst, src, rel))
        dst, src, rel = np.array(edges).transpose()

    graph = Graph(num_nodes, num_rels, np.stack((src, rel, dst)))
    norm = graph.compute_degree_norm()

    return graph, rel, norm


def get_adj_and_degrees(num_nodes: int, triplets: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    adj_list = [[] for _ in range(num_nodes)]

    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def sample_edge_neighborhood(adj_list: np.ndarray,
                             degrees: np.ndarray,
                             n_triplets: int,
                             sample_size: int) -> np.ndarray:
    """ Edge neighborhood sampling to reduce training graph size
    """
    edges = np.zeros(sample_size, dtype=np.int32)

    # initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = weights / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def generate_sampled_graph_and_labels(triplets: np.ndarray,
                                      sample_size: int,
                                      split_size: int,
                                      num_rels: int,
                                      adj_list: List[np.ndarray],
                                      degrees: np.ndarray,
                                      negative_rate: float):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets),
                                     sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels, (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def build_test_graph(num_nodes: int, num_rels: int, edges: np.ndarray, inverse: bool = False):
    """Inverse indicates whether to add inverted edges.
    """
    return build_graph_from_triplets(num_nodes, num_rels, edges.T, inverse)
