import torch
import torch.nn.functional as F
from torch.nn import GRUCell
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.utils import add_self_loops


class InteractionNet(MessagePassing):
    def __init__(
        self,
        node_features: int,
        add_self_loops: bool = False,
        aggr: str = "max",
        **kwargs
    ):
        # `aggr` goes to super(), not just onto self. Until torch_geometric 2.1 the
        # aggregation was read off `self.aggr` when `aggregate` ran, so assigning it after
        # super() worked. 2.1 moved aggregation into an `aggr_module` built during
        # `__init__`, and a later assignment to `self.aggr` no longer reaches it: this layer
        # kept reporting `self.aggr == "max"` while summing.
        #
        # That is the whole of the vdW divergence. Three messages of 1, 5, 3 aggregate to 5
        # on 2.0.3 and to 9 on 2.3.1, which changes the node embeddings — and the Morse vdW
        # term is the only energy that reads them, so the other four matched exactly and the
        # fault looked like it had to be somewhere else. Passing it here behaves identically
        # on 2.0.3, so one source tree serves both tiers.
                # "add", not "sum": torch_geometric 2.0.3 asserts the name is one of
        # add/mean/max/None and rejects "sum" outright, while every later release takes
        # "add" as the alias for SumAggregation. One spelling that both accept.
        super().__init__(aggr="add" if aggr == "sum" else aggr, **kwargs)
        self.W1 = Linear(node_features, node_features)
        self.W2 = Linear(node_features, node_features)
        # GRUCell(input_size, hidden_size) -> hidden_size
        self.rnn = GRUCell(node_features, node_features)
        self.add_self_loops = add_self_loops
        self.aggr = aggr

    def forward(self, x, edge_index):
        # Need to pass num_nodes to handle isolated nodes (in proteins).
        num_nodes = x.size(0)

        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        x_updated = self.propagate(edge_index, x=x, size=(num_nodes, num_nodes))
        return x_updated

    def message(self, x_j):
        return self.W2(x_j)

    def update(self, inputs, x):
        x_prime = F.relu(self.W1(x) + inputs)
        return self.rnn(x_prime, x)

    def compare(self, x, sample):
        """\
        Temporary function to compare the `forward` implementation.
        """

        def get_neighbors(sample, idx: int, order: int):
            edges = sample.edge_index_c
            if self.add_self_loops:
                edges, _ = add_self_loops(edges, num_nodes=sample.x.size(0))

            ligand_size = sample.is_ligand.sum().item()

            # If `idx` is of ligand,
            if order == 0:
                srcs = edges[0, edges[1] == idx]
                srcs = srcs - ligand_size
            # If `idx` is of target,
            elif order == 1:
                srcs = edges[0, edges[1] == idx + ligand_size]
            return srcs

        x1 = x[sample.is_ligand]
        x2 = x[~sample.is_ligand]

        # ligand <- protein
        M = self.W2(x2)
        A = torch.zeros(x1.size(0), M.size(1))
        for i in range(x1.size(0)):
            srcs = get_neighbors(sample, i, 0)
            # If isolated,
            if not srcs.numel():
                continue
            A[i] = torch.max(M[srcs], 0).values
        x_prime = F.relu(self.W1(x1) + A)
        x1_updated = self.rnn(x_prime, x1)

        # protein <- ligand
        M = self.W2(x1)
        A = torch.zeros(x2.size(0), M.size(1))
        for i in range(x2.size(0)):
            srcs = get_neighbors(sample, i, 1)
            # If isolated,
            if not srcs.numel():
                continue
            A[i] = torch.max(M[srcs], 0).values
        x_prime = F.relu(self.W1(x2) + A)
        x2_updated = self.rnn(x_prime, x2)

        return torch.cat((x1_updated, x2_updated), 0)
