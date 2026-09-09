"""Train PIGNet2 on affinity, or fine-tune it from a transferred encoder.

The authors' `src/exe/train.py` is Hydra-driven, multi-task, TensorBoard-logging, and reads
preprocessed pickles listed by key files. None of that survives contact with the harness, which
hands a trainer a directory of complexes and a csv of labels. **Their loss and their optimiser
do survive, and are used unchanged:**

    loss = MSE(sum of energy terms, y) + loss_dvdw_ratio * mean(dvdw_radii^2)

read straight off `PIGNet.loss_regression` and `PIGNet.loss_dvdw`, weighted by
`config.run.loss_dvdw_ratio` exactly as `training_step` does, with the optimiser from their own
`configure_optimizers`. What is dropped is the multi-task machinery — docking, screening and
derivative samples are separate datasets we do not have — so this trains the scoring task
alone and says so in `summary.json`.

**Labels are converted, not passed through.** Their scoring dataset uses
`label = pK * -1.36` (`src/data/data.py:351`): a binding free energy in kcal/mol, negative for
tight binders, while the benchmark's csv holds pK. Training on pK directly would fit the model
to predict the negative of what its physics computes, and it would still converge — to a model
whose vdW term has the wrong sign.

**Featurisation is the expensive part** — their pipeline protonates the protein with pymol and
the ligand with dimorphite, per complex — so prepared graphs are cached under `--cache` and
reused. That is also why `--cache` matters more here than for any other model in the roster.

usage:
    python train_complexes.py --complexes /data --labels /splits/train.csv \
        --val-labels /splits/val.csv --config-from /ckpt/pda_0.pt --out /outputs \
        --cache /cache --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import RDLogger
from tqdm import tqdm

for sub in ("src", "src/exe", "dataset/preprocess"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), sub))

import torch_geometric.data as pyg_data  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from predict import read_data  # noqa: E402
from transfer import is_head, set_training_mode, transfer_encoder  # noqa: E402

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

#: pK -> kcal/mol, the authors' own constant from src/data/data.py:351
KCAL_PER_PK = -1.36

PROTONATE = True
"""The authors' protonation step, as run_complexes.py runs it for inference.

Training and inference must featurise identically or the model is fitted to one distribution
and scored on another — which would look like a modest, plausible loss of accuracy.
"""


#: Interpolations only Hydra can resolve. Their config carries `run.log_file:
#: ${hydra:job.name}.log` and `experiment_name: ${now:...}`, which raise
#: `UnsupportedInterpolationType` the moment anything outside Hydra resolves the config —
#: including writing it back out. They are logging plumbing, so they are dropped rather than
#: worked around.
HYDRA_ONLY = ("${hydra:", "${now:")


def strip_hydra_interpolations(node):
    """Drop leaves whose value only Hydra could resolve. Recursive over dicts and lists."""
    if isinstance(node, dict):
        return {k: strip_hydra_interpolations(v) for k, v in node.items()
                if not (isinstance(v, str) and any(m in v for m in HYDRA_ONLY))}
    if isinstance(node, list):
        return [strip_hydra_interpolations(v) for v in node]
    return node


def config_to_yaml(config) -> str:
    """Serialise the config as a plain yaml string, for a checkpoint that opens strictly.

    Saving the omegaconf object itself is what makes the authors' checkpoints refuse
    `weights_only=True` and need laundering, so this writes text instead. `resolve=False`
    throughout: resolving is what raises on the Hydra-only keys, and every value we actually
    read back — lr, batch_size, conv_range, the model block — is a literal already.
    """
    container = strip_hydra_interpolations(OmegaConf.to_container(config, resolve=False))
    return OmegaConf.to_yaml(OmegaConf.create(container))


def featurise(complexes: str, labels: dict[str, float], conv_range) -> list:
    """Complexes to PyG Data, with the affinity attached in the authors' units."""
    data = []
    failed = []
    for cid in tqdm(sorted(labels), desc="Featurising"):
        d = Path(complexes) / cid
        if not d.is_dir():
            failed.append(f"{cid}: no directory")
            continue
        try:
            datum, attempts = None, []
            # Both ligand formats, as run_complexes.py does: some PDBbind SDFs do not
            # sanitise, and read_mols then yields None several frames from the cause.
            for name in (f"{cid}_ligand.mol2", f"{cid}_ligand.sdf"):
                if not (d / name).exists():
                    continue
                try:
                    read = read_data(d / f"{cid}_protein.pdb", d / name, conv_range,
                                     protonate_sdf=PROTONATE, protonate_protein=PROTONATE)
                except Exception as inner:
                    attempts.append(f"{name}: {type(inner).__name__}: {inner}")
                    continue
                datum = next((x for x, _n in read if x is not None), None)
                if datum is not None:
                    break
            if datum is None:
                raise ValueError("no readable ligand; " + (" | ".join(attempts) or "no files"))
            datum.y = torch.tensor([[labels[cid] * KCAL_PER_PK]], dtype=torch.float)
            datum.complex_id = cid
            data.append(datum)
        except Exception as e:
            failed.append(f"{cid}: {type(e).__name__}: {e}")

    if failed:
        print(f"{len(failed)} complexes failed to featurise:")
        for line in failed[:10]:
            print("  ", line)
    return data


def load_or_build(cache: str | None, complexes: str, labels_path: str, conv_range) -> list:
    """Featurise, or reuse a cached copy keyed by the split and the convolution range.

    `weights_only=False` on the way back in: these are PyG `Data` objects, which are not plain
    tensors, and the file was written by this script in this container rather than downloaded.
    That is the whole difference from the rule about foreign checkpoints.
    """
    with open(labels_path) as f:
        labels = {r["complex_id"]: float(r["y_true"]) for r in csv.DictReader(f)}

    if cache:
        stem = Path(labels_path).stem
        path = Path(cache) / f"pignet2_{stem}_conv{conv_range[0]:g}-{conv_range[1]:g}.pt"
        if path.exists():
            data = torch.load(path, weights_only=False)
            print(f"reusing {path}: {len(data)} graphs")
            return data
        data = featurise(complexes, labels, conv_range)
        torch.save(data, path)
        print(f"cached {len(data)} graphs -> {path}")
        return data

    return featurise(complexes, labels, conv_range)


def run_epoch(model, data, batch_size, device, optimiser, dvdw_ratio, frozen) -> dict:
    """One pass. `optimiser=None` means validation.

    The loss is `training_step`'s, restricted to a single regression task: their multi-task
    weighting is `loss_energy * loss_ratio + loss_dvdw * loss_dvdw_ratio` summed over tasks,
    and with one task at ratio 1 that reduces to what is written here.
    """
    training = optimiser is not None
    set_training_mode(model, training, frozen_encoder=frozen)

    order = torch.randperm(len(data)) if training else torch.arange(len(data))
    total_energy, total_dvdw, n = 0.0, 0.0, 0
    preds, trues = [], []

    for start in range(0, len(order), batch_size):
        chunk = [data[i] for i in order[start:start + batch_size]]
        batch = pyg_data.Batch.from_data_list(chunk).to(device)
        with torch.set_grad_enabled(training):
            energies, dvdw_radii = model(batch)
            loss_energy = F.mse_loss(energies.sum(-1, True), batch.y)
            loss_dvdw = dvdw_radii.pow(2).mean()
            objective = loss_energy + dvdw_ratio * loss_dvdw
        if training:
            model.zero_grad()
            objective.backward()
            optimiser.step()

        total_energy += float(loss_energy) * len(chunk)
        total_dvdw += float(loss_dvdw) * len(chunk)
        n += len(chunk)
        preds.append(energies.sum(-1).detach().cpu().numpy().ravel())
        trues.append(batch.y.detach().cpu().numpy().ravel())

    p, t = np.concatenate(preds), np.concatenate(trues)
    r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and p.std() and t.std() else float("nan")
    return {"loss": total_energy / max(n, 1), "dvdw": total_dvdw / max(n, 1),
            "r": r, "rmse": float(np.sqrt(((p - t) ** 2).mean()))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--complexes", required=True)
    p.add_argument("--labels", required=True, help="csv with complex_id,y_true in pK")
    p.add_argument("--val-labels", dest="val_labels", default=None)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--config-from", dest="config_from", required=True,
                   help="a published checkpoint, read for its architecture config. PIGNet2 has "
                        "no config file outside Hydra, and the checkpoint carries the one that "
                        "was actually trained with.")
    p.add_argument("--init-weights", dest="init_weights", default=None,
                   help="start from this checkpoint's weights, not just its config")
    p.add_argument("--init-encoder", dest="init_encoder", default=None,
                   help="encoder .pt from `gnnb encoder split`")
    p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    p.add_argument("--cache", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=None, help="default: the authors' 64")
    p.add_argument("--lr", type=float, default=None, help="default: the authors' 4e-4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    import utils  # local: needs the sys.path above

    # `weights_only=False` is unavoidable for the authors' files — the config is an omegaconf
    # object, so the strict loader refuses it. This runs in a container with no network, and it
    # is the reason tools/launder_checkpoint.py exists. What this script *writes* is plain
    # tensors and a yaml string, so its own output does not inherit the problem.
    reference = torch.load(args.config_from, map_location=device, weights_only=False)
    config = utils.merge_configs(reference["config"], {})
    if args.lr is not None:
        config.run.lr = args.lr
    if args.batch_size is not None:
        config.run.batch_size = args.batch_size
    conv_range = tuple(config.model.conv_range)

    train = load_or_build(args.cache, args.complexes, args.labels, conv_range)
    if not train:
        raise SystemExit("no training graphs were built")
    val = (load_or_build(args.cache, args.complexes, args.val_labels, conv_range)
           if args.val_labels else None)
    print(f"{len(train)} training graphs" + (f", {len(val)} validation" if val else ""))

    in_features = int(train[0].x.shape[1])
    if args.init_weights:
        weights = torch.load(args.init_weights, map_location=device, weights_only=False)
        model = utils.initialize_state(device, weights, config)[0]
        print(f"started from the published weights: {args.init_weights}")
    else:
        model = utils.initialize_state(device, None, config, in_features)[0]

    provenance: dict = {"init_encoder": None, "frozen_params": 0}
    if args.init_encoder:
        provenance = transfer_encoder(model, args.init_encoder, args.freeze_encoder)
    elif args.freeze_encoder:
        raise SystemExit("--freeze-encoder without --init-encoder would fit the energy function "
                         "against a random representation; that is not an experiment")
    model.to(device)

    optimiser = torch.optim.Adam(
        [q for q in model.parameters() if q.requires_grad],
        lr=config.run.lr, weight_decay=config.run.weight_decay)
    batch_size = int(config.run.batch_size)
    dvdw_ratio = float(config.run.loss_dvdw_ratio)
    print(f"lr {config.run.lr}, weight_decay {config.run.weight_decay}, batch {batch_size}, "
          f"loss_dvdw_ratio {dvdw_ratio}")

    out = Path(args.out)
    out_dir = out.parent if out.suffix == ".pt" else out
    model_path = out if out.suffix == ".pt" else out / "model.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best, best_epoch = [], float("inf"), -1
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train, batch_size, device, optimiser, dvdw_ratio,
                       args.freeze_encoder)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_r": tr["r"],
               "val_loss": "", "val_r": "",
               "train_rmse": tr["rmse"], "train_dvdw": tr["dvdw"], "val_rmse": ""}
        line = (f"epoch {epoch:4d}  train mse {tr['loss']:7.4f}  dvdw {tr['dvdw']:7.4f}  "
                f"R {tr['r']:+.3f}")
        if val:
            va = run_epoch(model, val, batch_size, device, None, dvdw_ratio, args.freeze_encoder)
            row.update(val_loss=va["loss"], val_r=va["r"], val_rmse=va["rmse"])
            line += f"  |  val mse {va['loss']:7.4f}  R {va['r']:+.3f}"
            score = va["loss"]
        else:
            score = tr["loss"]
        history.append(row)
        print(line, flush=True)

        if score < best:
            best, best_epoch = score, epoch
            # The config goes in as a yaml *string*, not an omegaconf object. Saving the object
            # is what makes the authors' own checkpoints refuse `weights_only=True` and need
            # laundering; there is no reason to inherit that.
            torch.save({"model_state_dict": model.state_dict(),
                        "in_features": in_features,
                        "config_yaml": config_to_yaml(config),
                        "epoch": epoch, "score": best}, model_path)

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "pignet2",
        "task": "scoring only; the authors' docking, screening and derivative tasks need "
                "datasets the harness does not hand a trainer",
        "label_units": "kcal/mol, pK * -1.36 as src/data/data.py:351",
        "epochs": len(history), "best_epoch": best_epoch, "best_score": best,
        "selected_on": "val_mse" if val else "train_mse",
        "train_graphs": len(train), "val_graphs": len(val) if val else 0,
        "seed": args.seed, "freeze_encoder": args.freeze_encoder,
        "lr": float(config.run.lr), "batch_size": batch_size,
        "loss_dvdw_ratio": dvdw_ratio,
        "seconds": round(time.time() - started, 1),
        **provenance,
    }, indent=2) + "\n")

    print(f"\nbest {best:.4f} at epoch {best_epoch} -> {model_path}  "
          f"({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
