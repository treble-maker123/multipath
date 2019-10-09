from typing import List, Tuple, Dict, Union

import torch.optim as optim

Triplets = Tuple[List[str], List[str], List[str]]
Vocabulary = Dict[str, int]
Path = Tuple[int]
Optimizer = Union[optim.Adam]
