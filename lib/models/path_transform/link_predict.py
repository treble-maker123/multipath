from pdb import set_trace

from torch import Tensor

from lib.models import Model
from lib.models.path_transform.embedding import EntityEmbedding, RelationEmbedding
from lib.utils import Graph
from multiprocessing import Pool, cpu_count


class LinkPredict(Model):
    def __init__(self, max_hops: int, num_entities: int, num_relations: int, hidden_dim: int,
                 num_transformer_layers: int):
        super().__init__()

        self.max_hops = max_hops
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        self.embed_entities = EntityEmbedding(num_entities, hidden_dim)
        self.embed_relations = RelationEmbedding(num_relations, hidden_dim)

    def forward(self, data: Tensor, graph: Graph) -> Tensor:
        for data_point in data:
            src_node_id, dst_node_id = data_point.unsqueeze(1)  # unsqueeze so src and dst node id would be size 1
            paths = Model.enumerate_paths(src_node_id, dst_node_id, self.max_hops, graph)
            set_trace()

    def loss(self, data: Tensor, labels: Tensor, graph: Graph) -> Tensor:
        path_embedding = self.forward(data, graph)
