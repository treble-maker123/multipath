from torch import Tensor


def check_all(tensor: Tensor):
    return count_nan(tensor) + count_inf(tensor) + too_large(tensor)


def count_nan(tensor: Tensor):
    total = (tensor != tensor).sum()
    if total > 0:
        print("FOUND NAN!")
    return total


def count_inf(tensor: Tensor):
    total = (tensor == float("inf")).sum() + (tensor == float("-inf")).sum()
    if total > 0:
        print("FOUND INF!")
    return total


def too_large(tensor: Tensor):
    total = (tensor > 1e10).sum()
    if total > 0:
        print("FOUND TOO LARGE!")
    return total
