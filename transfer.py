"""Move a trained encoder into a fresh PIGNet2, and freeze it if asked.

The harness cuts checkpoints (`gnnb encoder split`); this is the other half, which needs the
model class and so lives here.

**Where PIGNet2's boundary is.** The encoder is `embed` + `intraconv` + `interconv` —
everything producing node representations. Everything after it is an *energy function* rather
than an MLP: four networks mapping pair features to physical parameters, plus five learned
scalars setting the potential minima. Those scalars are one parameter each, so putting them on
the wrong side is invisible numerically while transferring task-specific physics as though it
were representation.

**PIGNet2 is the exception to "the encoder is exactly what produces the embedding".** Its
headline embedding is the four energy terms, and those are computed *by the head*. So the probe
number for `embeddings.npz` describes the physics head, not the transferable encoder — the one
that does predict what transfer moves is `embeddings.pooled.npz`, the summed node
representation, which is exactly what this file's encoder half produces. Read the two
accordingly; they are answers to different questions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The energy function, by parameter-name prefix. Must agree with `encoder.head` for the
#: pignet2 variants in harness/registry.toml — the harness cuts the file, this loads it, and a
#: disagreement produces tensors belonging to neither side.
HEAD_PREFIXES = ("nn_vdw_epsilon", "nn_vdw_width", "nn_vdw_radius", "nn_dvdw",
                 "hbond_coeff", "hydrophobic_coeff", "metal_ligand_coeff", "ionic_coeff",
                 "rotor_coeff")


def is_head(name: str) -> bool:
    """Matched at module boundaries so a prefix cannot claim a longer name that starts with it."""
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def read_encoder(path: str) -> dict[str, torch.Tensor]:
    """Load an encoder file, accepting either shape it arrives in.

    `gnnb encoder split` writes `{"encoder": {...}, ...}`; a bare state dict turns up when
    someone points this at a whole checkpoint. `weights_only=True` throughout — an encoder file
    is tensors, and PIGNet2's *own* checkpoints are the reason `tools/launder_checkpoint.py`
    exists, so nothing here relaxes the strict loader.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("encoder"), dict):
        payload = payload["encoder"]
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a state dict, got {type(payload).__name__}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def transfer_encoder(model: nn.Module, path: str, freeze: bool = False) -> dict[str, object]:
    """Overlay an encoder onto a fresh model, leaving the energy function at its initialisation.

    Overlaid and loaded strictly rather than with `strict=False`: a partial load that silently
    moved nothing is indistinguishable from success until the curve disappoints weeks later.
    """
    encoder = read_encoder(path)
    target = model.state_dict()

    head_keys = [k for k in encoder if is_head(k)]
    if head_keys:
        raise SystemExit(
            f"{path} carries {len(head_keys)} head tensors ({head_keys[:3]}). That is a whole "
            f"checkpoint, not an encoder — cut it with `gnnb encoder split` first, or the "
            f"fine-tune starts from the old task's energy function."
        )
    unknown = [k for k in encoder if k not in target]
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} tensors have no slot in this model: {unknown[:5]}")
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if mismatched:
        raise SystemExit(
            f"{path}: shape mismatch on {len(mismatched)} tensors: "
            + ", ".join(f"{k} {tuple(encoder[k].shape)} into {tuple(target[k].shape)}"
                        for k in mismatched[:3])
            + ". Different width or a different in_features; rebuild it or match the config."
        )
    uncovered = [k for k in target if k not in encoder and not is_head(k)]
    if uncovered:
        raise SystemExit(
            f"{path}: {len(uncovered)} model tensors are neither in the encoder nor in the "
            f"head — the boundary is wrong: {uncovered[:5]}"
        )

    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")

    frozen = 0
    if freeze:
        for name, param in model.named_parameters():
            if not is_head(name):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze {frozen:,} encoder params; {trainable:,} trainable in the energy function")

    return {"init_encoder": path, "transferred_tensors": len(encoder),
            "transferred_params": moved, "frozen_params": frozen}


def set_training_mode(model: nn.Module, training: bool, frozen_encoder: bool) -> None:
    """Keep a frozen encoder deterministic: its dropout would resample the representation the
    head is being fitted on, which is a different experiment from the one `gnnb probe` measures.
    """
    if not training or not frozen_encoder:
        model.train(training)
        return
    model.eval()
    for name, module in model.named_modules():
        if name and is_head(name) and any(p.requires_grad for p in module.parameters()):
            module.train()
