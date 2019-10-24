import torch


def paths_stack_collate(batch):
    """Stack the paths along the first dimension instead of creating a new dimension, batch is of the shape
    List[(paths, masks, triplet, relation tensor, num paths)]
    """
    paths = torch.cat(list(map(lambda x: x[0], batch)))
    masks = torch.cat(list(map(lambda x: x[1], batch)))
    triplets = torch.stack(list(map(lambda x: x[2], batch)))
    relations = torch.stack(list(map(lambda x: x[3], batch)))
    num_paths = torch.stack(list(map(lambda x: x[4], batch)))

    return paths, masks, triplets, relations, num_paths
