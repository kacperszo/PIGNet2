"""Locate exactly where GatedGAT diverges between torch_geometric versions.

Background: PIGNet2 reproduces its golden file only with torch_geometric 2.0.3. Under 2.3.1
the van der Waals term moves from -2.074 to -3.518 and the model drops to the heavy-atom
baseline. vdW is the only energy that reads the node embeddings GatedGAT produces, so the
divergence is somewhere in this layer.

Knowing *which line* moved is what makes the modernisation possible: the goal is a GatedGAT
that gives 2.0.3's numbers on a current PyG, rather than a permanent pin to a release that
has no CUDA wheels left.

Run the same file in both images and diff the printed tensors:

    podman run --rm --network=none -v $PWD:/w:ro localhost/pignet2-ref:latest \
        python /w/probe_gatedgat.py > ref.txt
    podman run --rm --network=none -v $PWD:/w:ro localhost/pignet2:latest \
        python /w/probe_gatedgat.py > new.txt
    diff ref.txt new.txt

Everything is seeded and weights are set explicitly, so any difference is the library's.
"""

import torch
import torch_geometric
from torch_geometric.utils import add_self_loops, softmax

from src.models.layers.gated_gat import GatedGAT

torch.manual_seed(0)

IN, OUT, N = 6, 6, 5


def fixed_layer():
    """A GatedGAT whose every parameter is deterministic and version-independent."""
    layer = GatedGAT(IN, OUT)
    # force lazy Linear to materialise before weights are assigned
    layer(torch.zeros(N, IN), torch.tensor([[0], [1]]))
    with torch.no_grad():
        # Seed per parameter by its position in sorted order, never by hash(): Python
        # randomises string hashing per process, so a hash-derived seed makes the two
        # containers disagree on the weights and every later stage looks like a library
        # difference when it is only PYTHONHASHSEED.
        for i, (name, p) in enumerate(sorted(layer.named_parameters())):
            g = torch.Generator().manual_seed(1000 + i)
            p.copy_((torch.rand(p.shape, generator=g) - 0.5))
    return layer.eval()


def main() -> None:
    print("torch_geometric", torch_geometric.__version__, "| torch", torch.__version__)

    layer = fixed_layer()
    x = torch.linspace(-1, 1, N * IN).reshape(N, IN)
    # a deliberately irregular graph: node 4 is isolated, which is the case the authors' note
    # in `forward` about proteins is guarding against
    edge_index = torch.tensor([[0, 1, 2, 0, 3], [1, 0, 1, 2, 2]])

    print("\n--- parameters, so a shape or naming change cannot hide as a value change ---")
    for name, p in sorted(layer.named_parameters()):
        print(f"  {name:<16} {tuple(p.shape)}  sum={float(p.sum()):+.6f}")

    print("\n--- aggregation as the layer actually resolves it ---")
    print("self.aggr           :", getattr(layer, "aggr", None))
    mod = getattr(layer, "aggr_module", None)
    print("self.aggr_module    :", type(mod).__name__ if mod is not None else None)

    ei, _ = add_self_loops(edge_index, num_nodes=N)
    Wx = layer.W1(x)

    print("\n--- stage 1: raw attention logits E ---")
    E = torch.einsum("ei,ij,ej->e", Wx[ei[1]], layer.W2, Wx[ei[0]])
    E = E + torch.einsum("ei,ij,ej->e", Wx[ei[0]], layer.W2, Wx[ei[1]])
    print(torch.round(E, decimals=6).tolist())

    print("\n--- stage 2: softmax over the target index ---")
    A = softmax(E, ei[1], dim=0, num_nodes=N)
    print(torch.round(A, decimals=6).tolist())
    # A softmax that normalises over the wrong axis still sums to something; check the axis
    # the layer actually intends, one group per target node.
    sums = torch.zeros(N).scatter_add_(0, ei[1], A)
    print("per-target sums     :", torch.round(sums, decimals=6).tolist())

    print("\n--- stage 3: full forward ---")
    out = layer(x, edge_index)
    print(torch.round(out, decimals=6).tolist())

    print("\n--- stage 4: the authors' own reference implementation ---")
    # `compare` is their hand-written dense version of the same maths, kept in the repo. If
    # forward and compare agree under one version and disagree under the other, the library
    # changed the meaning of the message-passing path rather than the arithmetic.
    ref = layer.compare(x, edge_index)
    print(torch.round(ref, decimals=6).tolist())
    print("forward vs compare max abs diff:",
          float((out - ref).abs().max()))


if __name__ == "__main__":
    main()
