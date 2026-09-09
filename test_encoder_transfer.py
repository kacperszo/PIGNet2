"""Can PIGNet2's encoder be moved to a different head?

The harder case than IGN, and that is the point of running it. IGN's head is an MLP, so a
contract that works there might work only for MLP heads. PIGNet2's head is an **energy
function**: four networks mapping pair features to physical parameters, plus five learned
scalars setting the potential minima. If the boundary holds here it holds broadly.

Encoder is `embed` + `intraconv` + `interconv` — everything producing node representations.
Everything else is the energy function, including those five scalars. They are one parameter
each, so putting them on the wrong side is invisible numerically while transferring
task-specific physics as though it were representation.

usage: python test_encoder_transfer.py
"""

import os
import sys
import warnings

import torch
import torch.nn as nn

for sub in ("", "src", "src/exe", "dataset/preprocess"):
    sys.path.insert(0, os.path.join("/work", sub))
warnings.filterwarnings("ignore")

import utils  # noqa: E402

from transfer import HEAD_PREFIXES, is_head  # noqa: E402

CHECKPOINT = "/ckpt/pda_0.pt"
# Read from transfer.py rather than restated, so the boundary this test checks and the boundary
# the trainer transfers on cannot drift apart — which is the failure the test exists to catch.
HEAD = list(HEAD_PREFIXES)


def build():
    """Rebuild exactly as their predict.py does, from the config the checkpoint carries."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = utils.merge_configs(checkpoint["config"], {})
    model = utils.initialize_state("cpu", checkpoint, config)[0]
    return model, checkpoint["model_state_dict"], config


def main() -> None:
    fresh, trained, config = build()

    encoder = {k: v for k, v in trained.items() if not is_head(k)}
    head = {k: v for k, v in trained.items() if is_head(k)}
    e = sum(v.numel() for v in encoder.values())
    h = sum(v.numel() for v in head.values())
    print("split: encoder {} tensors / {:,} params ({:.0f}%) | head {} / {:,}".format(
        len(encoder), e, 100 * e / (e + h), len(head), h))

    # a genuinely fresh model, so "loaded" means something
    model = utils.initialize_state("cpu", {"model_state_dict": {}}, config)[0] \
        if False else build()[0]
    for name, param in model.named_parameters():
        if not is_head(name):
            nn.init.normal_(param, std=0.5) if param.dim() > 0 else None

    # Overlay, not a partial load: PyG's lazy Linear raises KeyError from its load hook when
    # a head tensor is absent, and overlaying lets the load stay strict.
    target = model.state_dict()
    assert all(k in target for k in encoder), "encoder keys have no slot in the model"
    model.load_state_dict({**target, **encoder}, strict=True)

    after = model.state_dict()
    exact = sum(1 for k in encoder if torch.equal(after[k], encoder[k]))
    print("1. encoder loaded : {}/{} tensors match the checkpoint exactly".format(
        exact, len(encoder)))
    assert exact == len(encoder)

    # 2. the energy function is replaced wholesale by something that is not physics
    class PlainHead(nn.Module):
        """Not an energy function at all — five outputs from pooled node features."""

        def __init__(self, dim: int, out: int = 5):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, out))

        def forward(self, x):
            return self.net(x)

    # PyG's Linear exposes no in_features/out_features; read the shape off the weight
    embed_w = model.state_dict()["embed.weight"]
    out_dim, in_dim = embed_w.shape
    new_head = PlainHead(out_dim)
    print("2. new head built : {} -> 5 outputs, replacing an energy function".format(out_dim))

    # 3. the encoder runs on its own and feeds it
    n_atoms = 30
    x = torch.randn(n_atoms, in_dim)
    edge_index = torch.randint(0, n_atoms, (2, 90))
    h_nodes = model.conv(model.embed(x), edge_index, edge_index)
    pooled = h_nodes.sum(dim=0, keepdim=True)
    out = new_head(pooled)
    print("3. encoder feeds  : nodes {} -> pooled {} -> out {}".format(
        tuple(h_nodes.shape), tuple(pooled.shape), tuple(out.shape)))
    assert out.shape == (1, 5)

    # 4. gradients reach the encoder
    model.zero_grad()
    new_head.zero_grad()
    out.sum().backward()
    enc_params = [p for n, p in model.named_parameters() if not is_head(n)]
    got = [p for p in enc_params if p.grad is not None and p.grad.abs().sum() > 0]
    print("4. gradients flow : {}/{} encoder tensors received a non-zero gradient".format(
        len(got), len(enc_params)))
    assert got

    print("\nPASS - the encoder detaches from the energy function and trains under a plain head.")


if __name__ == "__main__":
    main()
