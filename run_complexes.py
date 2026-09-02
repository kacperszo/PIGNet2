"""Score and embed a directory of complexes with PIGNet2, in one pass.

PIGNet2 is physics-informed, and that shapes both capabilities. Its `forward` does not end in
an MLP: after the graph convolutions it computes pairwise van der Waals, hydrogen-bond, metal
and hydrophobic terms, and sums them per complex (`pignet.py:161`). So

    prediction = sum of the energy terms
    embedding  = the terms themselves

The four numbers are the authors' own complex-level summary — `utils.py:191` labels them
`[vdw, hbond, ml, hydro]` — reached by their own aggregation. That makes them native in the
sense that has mattered elsewhere in this benchmark, where borrowing the authors' aggregation
beat inventing one every time.

But four dimensions is very little to transfer, so the pooled node representation is written
alongside for comparison, exactly as GenScore's three variants were:

    energies    4 dims, the physics bottleneck, native
    pooled      sum over atom embeddings after the convolutions, higher-dimensional

Which is worth more is an empirical question; the probe answers it.

usage:
    python run_complexes.py --complexes /data --model /ckpt/pda_0.pt --out /outputs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger
from tqdm import tqdm

for sub in ("src", "src/exe", "dataset/preprocess"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), sub))

import torch_geometric.data as pyg_data  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from predict import read_data  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


PROTONATE = True
"""Run the authors' protonation step, as they intended.

Enabled by the DimorphiteDL shim in sitecustomize.py — without it the current dimorphite-dl
breaks their pipeline, ours and their own README example alike.
"""



def load_model(path: str, device: str):
    """Rebuild the model exactly as their predict.py does.

    The checkpoint carries its own config, and `utils.initialize_state` is what turns the two
    into a model — reusing it rather than reconstructing the architecture by hand is what
    keeps this faithful.

    `weights_only=True` is not possible here: the config is an omegaconf object, so the
    pickle references `omegaconf.dictconfig.DictConfig` and the strict loader refuses it. The
    scan showed no code-execution primitive, and this runs inside the container with no
    network — but it is the reason `tools/launder_checkpoint.py` exists, and a laundered copy
    should be used once the config has been read out once.
    """
    import utils  # local import: needs sys.path set above

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = utils.merge_configs(checkpoint["config"], {})
    model = utils.initialize_state(device, checkpoint, config)[0]
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and embed with PIGNet2")
    parser.add_argument("--complexes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.model, device)
    conv_range = tuple(config.model.conv_range)

    # Capture the node representations the physics reads. `model.conv` is a *method* that
    # loops over self.intraconv then self.interconv, so there is nothing to hook there; the
    # last interconv layer is the real end of the encoder.
    captured: list[torch.Tensor] = []
    model.interconv[-1].register_forward_hook(lambda m, i, o: captured.append(o.detach()))

    ids = sorted(d for d in os.listdir(args.complexes)
                 if os.path.isdir(os.path.join(args.complexes, d)))
    os.makedirs(args.out, exist_ok=True)

    rows, energies, pooled, kept, failed = [], [], [], [], []
    for cid in tqdm(ids, desc="Scoring"):
        d = Path(args.complexes) / cid
        try:
            # Try both ligand formats. Some PDBbind SDFs do not sanitise — 1a30 among them —
            # and their read_mols then yields None, which surfaces much later as
            # MolToSmiles(NoneType) inside protonation rather than as a read failure.
            datum, attempts = None, []
            for name in (f"{cid}_ligand.mol2", f"{cid}_ligand.sdf"):
                if not (d / name).exists():
                    continue
                try:
                    data = read_data(d / f"{cid}_protein.pdb", d / name, conv_range,
                                     protonate_sdf=PROTONATE, protonate_protein=PROTONATE)
                except Exception as inner:
                    # keep the first real reason rather than reporting a generic
                    # "unreadable" for every complex, which hides the actual fault
                    attempts.append(f"{name}: {type(inner).__name__}: {inner}")
                    continue
                datum = next((x for x, _name in data if x is not None), None)
                if datum is not None:
                    break
            if datum is None:
                raise ValueError("no readable ligand; " + " | ".join(attempts) or "no files")

            batch = pyg_data.Batch.from_data_list([datum]).to(device)
            captured.clear()
            with torch.no_grad():
                terms, _dvdw = model(batch)

            terms = terms.squeeze(0).cpu().numpy()
            rows.append({"complex_id": cid, "y_pred": float(terms.sum())})
            energies.append(terms)
            # sum over atoms, matching how the energies themselves are aggregated
            pooled.append(captured[-1].sum(dim=0).cpu().numpy())
            kept.append(cid)
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")

    if failed:
        print(f"\n{len(failed)} complexes failed:")
        for f in failed[:10]:
            print("  ", f)
    if not kept:
        raise SystemExit("nothing scored")

    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["complex_id", "y_pred"])
        writer.writeheader()
        writer.writerows(rows)

    np.savez(os.path.join(args.out, "embeddings.npz"),
             ids=np.array(kept), vectors=np.stack(energies))
    np.savez(os.path.join(args.out, "embeddings.pooled.npz"),
             ids=np.array(kept), vectors=np.stack(pooled))

    print(f"\n{len(kept)} complexes")
    print(f"  energies {np.stack(energies).shape[1]} dims, pooled {np.stack(pooled).shape[1]} dims")


if __name__ == "__main__":
    main()
