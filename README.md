<!-- gnn-benchmark:begin -->
# Running this in gnn-benchmark

Two tiers off one source tree, both exact. **Use `pignet2.torch`**: the torch_geometric ceiling
that forced the 2.0.3 pin is gone, so it runs torch 2.5.1+cu124 with PyG 2.8 unpinned and no
torch-scatter at all.

| variant | stack | `gnnb verify` on CASF-2016 |
|---|---|---|
| `pignet2.reference` | torch 2.1.2, PyG 2.0.3 | 285/285, max abs diff 0 |
| `pignet2.torch` | torch 2.5.1+cu124, PyG 2.8 | 285/285, **max abs diff 0** |

The vdW term used to read -3.518 against the authors' committed -2.074 on anything newer than
PyG 2.0.3, with the other four energies exact. That was a one-line bug here, not the framework:
`InteractionNet` never passed `aggr` to `MessagePassing.__init__`, and from PyG 2.1 the later
assignment to `self.aggr` no longer reaches the aggregation — the layer reported max and summed.

```bash
podman build --format=docker -f Containerfile.torch -t pignet2-torch:latest .

gnnb verify --variant pignet2.torch --dataset data/CASF-2016/coreset
gnnb run --variant pignet2.torch --capability predict --dataset <complexes> --gpu
gnnb run --variant pignet2.torch --capability embed   --dataset <complexes>
```

The model is physics-informed, so the head is an energy function rather than an MLP: `predict`
is the sum of the four terms and `embed` is the terms themselves, with a pooled node
representation written beside them because four dimensions is very little to transfer.

The pins that matter, the dimorphite shim, and the `rdkit-pypi` pin that silently never took
are all in [CLAUDE.md](CLAUDE.md).

## Running it without the harness

This fork runs on its own; the benchmark adds bookkeeping, not capability. Every
command below is generated from the adapter by `gnnb howto`, so it cannot drift from
what the harness actually runs — regenerate with `python tools/sync_model_readmes.py`.

All of them run with `--network=none` and a read-only root filesystem. Nothing is
fetched at run time; dependencies are resolved when the image is built.

### What it eats

One directory per complex, named after it:

    <complexes>/<id>/<id>_protein.pdb
    <complexes>/<id>/<id>_ligand.sdf      # or .mol2; several models try both

Protein and ligand only, but both are protonated on the way in — the protein with PyMOL, the ligand with dimorphite-dl — so the container needs a writable working directory, which is why every command below starts with `mkdir -p /tmp/work && cd /tmp/work`.

### Build

```bash
podman build --format=docker -f Containerfile.authors -t pignet2-ref:latest .  # torch_geometric 2.0.3
podman build --format=docker -f Containerfile.torch   -t pignet2-torch:latest . # clean stack
```

### Run

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

### What comes out

| file | holds |
|---|---|
| `predictions.csv` | `complex_id,y_pred` — a binding free energy in kcal/mol, so it correlates *negatively* with pK. The adapter flips the sign; if you run this directly, you have to |
| `embeddings.npz` | `ids` and `vectors`, the physical energy terms the model sums |
| `embeddings.pooled.npz` | the summed node representation — the one that corresponds to the transferable encoder |
| `model.pt` | training only, and unlike the authors' it opens under `weights_only=True`: the config goes in as a YAML string rather than an omegaconf object |

### Before you trust the numbers

**Which embedding you want depends on the question.** The headline one is the energy terms, and those are computed by the *head* — so a probe on them describes the physics, not the representation a fine-tune would start from. For transfer, `embeddings.pooled.npz` is the one that matches the encoder boundary.

**The authors' checkpoints carry a Hydra config**, so they will not open under `weights_only=True`. `tools/launder_checkpoint.py` in the benchmark strips one to tensors inside a container; the trainer here writes files that need no such step.

<!-- gnn-benchmark:end -->

---

# PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening
This repository is the official implementation of [PIGNet2: A versatile deep learning-based protein-ligand interaction prediction model for accurate binding affinity scoring and virtual screening](https://arxiv.org/abs/2307.01066).

## Installation
You can download this repository by `git clone https://github.com/mseok/PIGNet2.git`. Then, you can proceed with the following steps.

## Requirements
### Environment Setup
You can use `conda` or `venv` for environment setting.
For the case of using `conda`, create the environment named `pignet2` as following.
```console
conda create -n pignet2 python=3.9
conda activate pignet2
conda install rdkit=2022.03.4 openbabel pymol-open-source -c conda-forge
```

### Install Dependencies
```console
pip install -r requirements.txt
```

## Data

Donwload our source data into `dataset` directory in this repository.
By executing `dataset/download.sh`, you can download all the following datasets.
> training dataset
- PDBbind v2020 scoring
- PDBbind v2020 docking
- PDBbind v2020 cross
- PDBbind v2020 random
> benchmark dataset
- CASF-2016 socring
- CASF-2016 docking
- CASF-2016 screening
- DUD-E screening
- derivative benchmark

Then, you can extract the downloaded files by executing `dataset/untar.sh`.

## Training
Training scripts can be found in `experiments/training_scripts` directory.
We provide 4 scripts for training.
- `baseline.sh`: training without any data augmentation
- `only_nda.sh`: training only with negative data augmentation
- `only_pda.sh`: training only with positive data augmentation
- `pda_nda.sh`: training with both positive and negative data augmentation

If you execute the script, the result files will be generated in your **current working directory**.
By default, we recommend you to execute training scripts at `experiemnts` directory.
All the result files are placed in `outputs/${EXPERIMENT_NAME}` directory.

## Benchmark
Benchmark scripts can be found in `experiments/benchmark_scripts` directory.
We provide 5 scripts for benchmark.
- `casf2016_scoring.sh`: benchmark on CASF-2016 scoring benchmark
- `casf2016_docking.sh`: benchmark on CASF-2016 docking benchmark
- `casf2016_screening.sh`: benchmark on CASF-2016 screening benchmark
- `dude.sh`: benchmark on DUD-E benchmark
- `derivative.sh`: benchmark on derivative benchmark (2015)

After training, you have to set the `${BENCHMARK_DIR}` in each benchmark scripts, which is set as `experiments/outputs/${EXPERIMENT_NAME}` as default.
Since `experiments/outputs` is set as a root directory of each experiment, it is highly recommended to place the `outputs` directory inside `experiments` directory.
For using our pre-trained model for benchmark, please refer to the [next section](#pre-trained-models).

After that, you will get the benchmark result files in `experiments/outputs/${EXPERIMENT_NAME}/benchmark`.
To benchmark each result files, you can execute `src/benchmark/*.py`.
For example, you can perform DUD-E benchmark by the following command.
```console
src/benchmark/dude_screening_power.py -f experiments/outputs/${EXPERIMENT_NAME}/benchmark/result_dude_${EPOCH}.txt -v
```

## Pre-trained Models
You can find the pre-trained models in `src/ckpt`.
We provide PIGNet2 models trained with both positive and negative data augemntation, which is the best model.
You can execute the `experiments/benchmark/pretrained_*.sh` scripts to get the benchmark results of pre-trained models.
The scripts will generate result files in `experiments/pretrained`.

# Using PIGNet2 for a single data point
> [!NOTE]  
> We highly recommend to use SMINA-optimized ligand conformations and doing 4-model ensemble to get accurate results.

Prepare protein pdb file and ligand sdf.
Execute the following command to generate the result in `$OUTPUT` path (the output path is `predict.txt` by default):
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p $PROTEIN -l $LIGAND -o $OUTPUT
```
By default, each element of result are named as `$(basename $PROTEIN .pdb)_$(basename $LIGAND .sdf)_${idx}`, where `${idx}` is an index of ligand conformation.

## Case 1: a single pdb and a single sdf with one conformation
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand_single_conformation.sdf -o examples/case1.txt
```

## Case 2: a single pdb and a single sdf with multiple conformations
`src/exe/predict.py` automatically enumerates all conformations in ligand sdf.
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand1.sdf -o examples/case2.txt
```

## Case 3: a single pdb and multiple sdfs with multiple conformations
`src/exe/predict.py` automatically make protein-ligand pairs for a single pdb and all ligand sdfs.
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p examples/protein.pdb -l examples/ligand1.sdf examples/ligand2.sdf -o examples/case3.txt
```

## Case 4: multiple pdbs and multiple sdfs with multiple conformations
In this case, you should match the order of ligand and protein files and all of them sequentially.
For example, if you have `protein1-ligand1`, `protein1-ligand2`, `protein2-ligand3`, you should do like following:
```console
python src/exe/predict.py ./src/ckpt/pda_0.pt -p protein1.pdb protein1.pdb protein2.pdb -l ligand1.sdf ligand2.sdf ligand3.sdf
```

# Explanation about the results

The result file is a tab-separated file with the following columns:

```
protein_ligand_single_conformation_0    0.000   -3.990  -2.074  -1.021  0.000   -0.894  0.000
```

Each of the numeric columns corresponds to:

- True label (which is just set to 0.000 in inference)
- Total predicted binding affinity (= sum of the right-hand values)
- van der Waals energy
- hydrogen bond energy
- metal-ligand coordination energy
- hydrophobic energy
- dummy variable (please ignore this column)
