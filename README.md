# PIGNet2 — physics-informed scoring with a learned energy function

> A fork maintained for [gnn-benchmark](../../README.md). The authors' own README is
> kept as [README.upstream.md](README.upstream.md) for attribution and for their
> description of the method — **its build and run instructions are not current for
> this fork.**

## What it is

Graph convolutions produce node representations, and the head is not an MLP but an
**energy function**: pairwise van der Waals, hydrogen-bond, metal-ligand and hydrophobic terms summed
per complex, with learned scalars setting the potential minima. The output is a binding free energy in
kcal/mol, negative for tight binders, so it correlates *negatively* with pK by construction.

Two tiers share one source: `pignet2.reference` pins torch_geometric 2.0.3 as the fidelity baseline,
`pignet2.torch` runs 2.5.1 with no compiled extensions and reproduces it bit-for-bit.

## State

| | |
|---|---|
| CASF-2016 scoring | **R 0.759**, Spearman 0.747, n=285 (RMSE not comparable — kcal/mol) |
| embedding | the 5 energy terms, **native**, probe R 0.790 — 104% of its own head |
| `gnnb verify` | 285/285, **0.0**, both tiers |
| training | `train` and `finetune`, with the authors' loss and optimiser |

## Build

```bash
podman build --format=docker -f Containerfile.authors -t pignet2-ref:latest .  # torch_geometric 2.0.3
podman build --format=docker -f Containerfile.torch   -t pignet2-torch:latest . # clean stack
```

## Run it, without the harness

Generated from this model's adapter by `gnnb howto`, so these are the exact commands
the benchmark issues — regenerate with `python tools/sync_model_readmes.py`. Every one
runs with `--network=none` and a read-only root filesystem.

Input is one directory per complex:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Protein and ligand only, but both are protonated on the way in — the protein with PyMOL, the ligand with dimorphite-dl — so the container needs a writable working directory, which is why every command below starts with `mkdir -p /tmp/work && cd /tmp/work`.

```bash
# pignet2.reference — localhost/pignet2-ref:latest
# source: models/pignet2

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/src/ckpt:/ckpt:ro" \
    localhost/pignet2-ref:latest \
    sh -c 'mkdir -p /tmp/work && cd /tmp/work && python /work/run_complexes.py --complexes /data --model /ckpt/pda_0.pt --out /outputs --device cpu'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/src/ckpt:/ckpt:ro" \
    localhost/pignet2-ref:latest \
    sh -c 'mkdir -p /tmp/work && cd /tmp/work && python /work/run_complexes.py --complexes /data --model /ckpt/pda_0.pt --out /outputs --device cpu'
```

```bash
# pignet2.torch — localhost/pignet2-torch:latest
# source: models/pignet2

# predict
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/src/ckpt:/ckpt:ro" \
    localhost/pignet2-torch:latest \
    sh -c 'mkdir -p /tmp/work && cd /tmp/work && python /work/run_complexes.py --complexes /data --model /ckpt/pda_0.pt --out /outputs --device cpu'

# embed
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/src/ckpt:/ckpt:ro" \
    localhost/pignet2-torch:latest \
    sh -c 'mkdir -p /tmp/work && cd /tmp/work && python /work/run_complexes.py --complexes /data --model /ckpt/pda_0.pt --out /outputs --device cpu'

# train
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/src/ckpt:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/pignet2-torch:latest \
    sh -c 'mkdir -p /tmp/work && cd /tmp/work && python /work/train_complexes.py --complexes /data --labels /splits/train.csv --out /outputs --config-from /ckpt/pda_0.pt --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30'

# finetune  (encoder frozen; drop --freeze-encoder to tune all of it)
podman run --rm \
    --network=none --read-only \
    --tmpfs /tmp:rw,size=2g \
    -v /path/to/complexes:/data:ro \
    -v /path/to/outputs:/outputs:rw,U \
    -v "$PWD/src/ckpt:/ckpt:ro" \
    -v /path/to/splits:/splits:ro \
    -v /path/to/cache:/cache:rw,U \
    --shm-size 4g \
    localhost/pignet2-torch:latest \
    sh -c 'mkdir -p /tmp/work && cd /tmp/work && python /work/train_complexes.py --complexes /data --labels /splits/train.csv --out /outputs --config-from /ckpt/pda_0.pt --seed 0 --device cpu --val-labels /splits/val.csv --cache /cache --epochs 30 --init-encoder /ckpt/encoder.pt --freeze-encoder'
```

## What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` — a binding free energy in kcal/mol, so it correlates *negatively* with pK. The adapter flips the sign; if you run this directly, you have to |
| `embeddings.npz` | `ids` and `vectors`, the physical energy terms the model sums |
| `embeddings.pooled.npz` | the summed node representation — the one that corresponds to the transferable encoder |
| `model.pt` | training only, and unlike the authors' it opens under `weights_only=True`: the config goes in as a YAML string rather than an omegaconf object |

## Before you trust the numbers

**Which embedding you want depends on the question.** The headline one is the energy terms, and those are computed by the *head* — so a probe on them describes the physics, not the representation a fine-tune would start from. For transfer, `embeddings.pooled.npz` is the one that matches the encoder boundary.

**The authors' checkpoints carry a Hydra config**, so they will not open under `weights_only=True`. `tools/launder_checkpoint.py` in the benchmark strips one to tensors inside a container; the trainer here writes files that need no such step.

## Maintainer notes

`CLAUDE.md` in this directory holds what breaks if it is changed back.
