from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Tuple, Optional

import torch
from dgl.subgraph import DGLSubGraph
from torch import Tensor

from config import configured_device
from lib.models.module import Module
from lib.types import Path
from lib.utils import Graph


class Model(Module, ABC):
    def __init__(self):
        Module.__init__(self)
        ABC.__init__(self)
        self.device = configured_device

    @abstractmethod
    def forward(self, data: Tensor, graph: Graph, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def loss(self, data: Tensor, labels: Tensor, graph: Graph, **kwargs) -> Tensor:
        pass

    @classmethod
    def get_path_subgraph(cls,
                          src_node_id: Tensor,
                          dst_node_id: Tensor,
                          max_hops: int,
                          graph: Graph) -> DGLSubGraph:
        """Creates a subgraph consists of all paths going from src_node_id to dst_node_id, where src_node_id node
        contains only out-edges and dst_node_id node contains only in-edges.

        Args:
            src_node_id: Node ID of the source node
            dst_node_id: Node ID of the destination node
            max_hops: Maximum number of hops to make to identify paths between src_node_id and dst_node_id
            graph: The graph on which to perform this operation

        Returns: A subgraph containing all of the edges within max_hops away from src_node_id.

        """
        paths = cls.enumerate_paths(src_node_id, dst_node_id, max_hops, graph)
        edge_ids = list(set(chain(*map(cls._reduce_to_edges, paths))))
        return graph.edge_subgraph(edge_ids)

    @classmethod
    def enumerate_paths(cls,
                        src_node_id: Tensor,
                        dst_node_id: Tensor,
                        max_hops: int,
                        graph: Graph,
                        padding: Tuple[int, int] = None) -> Tuple[List[Path], Optional[List[Tensor]]]:
        """Enumerates all paths composed of edges start on src_node_id, and expanding breadth-first toward dst_node_id,
        with a maximum of max_hops.

        Args:
            src_node_id: A torch Tensor containing the source node ID
            dst_node_id: A torch Tensor containing the destination node ID
            max_hops: The most number of hops to make to identify paths
            graph: The graph on which to traverse
            padding: A tuple of indices representing the padding token for entity and relation
        """
        assert src_node_id.size() == torch.Size([1]) and dst_node_id.size() == torch.Size([1]), \
            f"Method only takes src_node_id and dst_node_id of 1, got {src_node_id.size()} and {dst_node_id.size()} " \
            f"instead."

        src_node_id, dst_node_id = src_node_id.cpu(), dst_node_id.cpu()
        hops = {
            0: [(src_node_id.item(),)]
        }
        frontier_nodes = src_node_id

        for nth_hop in range(max_hops):
            # filter out the frontier nodes that are also dst_node so that there are no out-edges from dst_node
            frontier_nodes = frontier_nodes[(frontier_nodes != dst_node_id).nonzero()].squeeze()

            # expand from frontier nodes
            hop_src_ids, hop_dst_ids, hop_edge_ids = graph.out_edges(frontier_nodes, form="all")
            num_triplets = len(hop_edge_ids)

            # To speed up processing, in the last hop, filter out the ids in hop_dst_ids that is not dst_node_id
            if nth_hop == max_hops - 1:
                matching_idx = (hop_dst_ids == dst_node_id).nonzero()
                matching_idx = matching_idx.squeeze(0) if len(matching_idx.shape) > 1 else matching_idx
                hop_src_ids = hop_src_ids[matching_idx]
                hop_dst_ids = hop_dst_ids[matching_idx]
                hop_edge_ids = hop_edge_ids[matching_idx]

            # convert the node and edge indices to entity and relation IDs
            src_entity_ids = graph.ndata["id"].squeeze()[hop_src_ids]
            rel_ids = graph.edata["type"][hop_edge_ids]
            dst_entity_ids = graph.ndata["id"].squeeze()[hop_dst_ids]

            arguments = [src_entity_ids, dst_entity_ids, rel_ids, [hops[nth_hop]] * num_triplets]
            hop_paths = list(map(cls._create_hop_paths, *arguments))

            hops[nth_hop + 1] = list(set(chain(*hop_paths)))
            frontier_nodes = hop_dst_ids

        # Filter out the paths that do not end with the target destination node
        candidate_paths = []
        for hop_idx, hop_paths in hops.items():
            if hop_idx == 0:  # only has itself
                continue

            target_paths = list(filter(lambda path: path[-1] == dst_node_id, hop_paths))
            candidate_paths += target_paths

        if padding is not None and len(candidate_paths) > 0:
            num_paths = len(candidate_paths)
            max_length = max(list(map(len, candidate_paths)))
            paths_and_masks = list(map(cls._pad_to_max_length,
                                       candidate_paths,
                                       [max_length] * num_paths,
                                       [padding] * num_paths))
            candidate_paths, masks = list(zip(*paths_and_masks))

            return candidate_paths, masks

        return candidate_paths, None

    @classmethod
    def _pad_to_max_length(cls,
                           path: Tuple[int],
                           max_length: int,
                           padding: Tuple[int, int]) -> Tuple[Tuple[int], Tensor]:
        num_to_pad = (max_length - len(path))
        # divide by 2 because extension consists of 2 elements
        padded_path = path + tuple(padding * (num_to_pad // 2))
        mask = torch.zeros(max_length, dtype=torch.bool)

        if num_to_pad > 0:
            mask[-num_to_pad:] = 1

        return padded_path, mask

    @staticmethod
    def _create_hop_paths(hop_src_id: Tensor,
                          hop_dst_id: Tensor,
                          hop_edge_id: Tensor,
                          paths_from_previous_hop: List[Path]) -> List[Path]:
        """Extend the paths from previous hop with the new (hop_src_id, hop_dst_id, hop_edge_id) path.
        """
        hop_edge_id, hop_dst_id = hop_edge_id.item(), hop_dst_id.item()
        return [(*previous_path, hop_edge_id, hop_dst_id)
                for previous_path in paths_from_previous_hop if previous_path[-1] == hop_src_id]

    @staticmethod
    def _reduce_to_edges(path: Path) -> List[int]:
        """Takes in a Path object, which is a tuple of ints of the format (e1, r1, e2, r2 ...) and obtain the relations
        (r1, r2 ...)
        """
        return [path[idx] for idx in range(len(path)) if idx % 2 == 1]
