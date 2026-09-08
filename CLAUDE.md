# CLAUDE.md — PIGNet2

## What this is

Physics-informed affinity prediction. `forward` does not end in an MLP: after the graph
convolutions it computes pairwise van der Waals, hydrogen-bond, metal, hydrophobic and ionic
terms and sums them per complex (`pignet.py:161`). The prediction is that sum; the terms
themselves are the model's own complex-level summary.

## Reproduced exactly, and how

The reference image matches the authors' committed golden file (`examples/case1.txt`) on
**all five energy terms and the sum**, to three decimals:

    -2.074  -1.021  0.000  -0.894  0.000   sum -3.990

Getting there took two independent fixes, and the model's construction is why the fault was
hard to see. `dev_vdw_radii_coeff` is 0, so the radii are a pure table lookup; the
hydrogen-bond, metal and hydrophobic terms take their minima from *scalar* learned
coefficients. The Morse vdW term is the **only** energy that reads the node embeddings the
convolutions produce — so four terms can match exactly while the features are wrong.

**The vdW term was wrong on anything newer than torch_geometric 2.0.3 — and the cause was a
one-line bug in this repo, not the environment.** With 2.3.1 the term read -3.518 against the
golden's -2.074 while the other four energies stayed exact.

`InteractionNet.__init__` called `super().__init__(**kwargs)` without passing `aggr`, then set
`self.aggr = "max"` afterwards. Until PyG 2.1 `aggregate()` read `self.aggr` when it ran, so
that worked; 2.1 moved aggregation into an `aggr_module` built during `__init__`, and the later
assignment stopped reaching it. The layer went on reporting `self.aggr == "max"` while summing.
Three messages of 1, 5, 3 aggregate to 5 on 2.0.3 and to 9 on 2.3.1.

The Morse vdW term is the only energy that reads node embeddings, which is why the other four
matched exactly and the fault looked like it had to be environmental. It was blamed on PyG's
MessagePassing rewrite, then — after a unit-level probe showed `GatedGAT` agreed across
versions — on rdkit. `GatedGAT` does agree; it was never the layer at fault, and the probe
never touched `InteractionNet`.

Passing `aggr` to `super()` fixes it, spelled `"add"` rather than `"sum"` because 2.0.3 asserts
the name is one of add/mean/max/None while later releases take "add" as the alias. Every tier
now reproduces `case1.txt`, and **the PyG pin is gone**: `Containerfile.torch` runs torch 2.5.1
with torch_geometric 2.8 unpinned and no torch-scatter at all. `gnnb verify pignet2.torch`
comes back **285/285 at max|Δ| = 0**: bit-exact, not merely close.

The experiment that settled it is worth keeping as a pattern: build the image that *works*,
change exactly one package on top of it, and rerun. The two earlier attempts compared images
differing in four packages at once, and the one that would have isolated it had a numpy 2
against a torch built for numpy 1, so it never produced a number.

**dimorphite-dl had to be shimmed.** The PyPI package was rewritten and no longer exposes the
`DimorphiteDL` class `protonate.py` calls, so protonation silently did not happen and the
hydrogen-bond term read -0.743 instead of -1.021. `sitecustomize.py` restores the class around
`mol.Protonate`, the engine that is still there and takes a plain dict rather than parsing
`sys.argv` the way the package's own importable entry points do.

**The image runs torch 2.1.2, not the authors' 1.9.1**, and only PyG carries their pin. That
is deliberate: 1.9.1 forces Python 3.9, dimorphite-dl publishes nothing for 3.9, and the
hydrogen-bond term would then be wrong instead of the vdW one. Neither the authors' full stack
nor a fully current one lets both fixes hold at once.

## Results on CASF-2016

| metric | value | note |
|---|---|---|
| Pearson | **0.759** | 0.490 before the fixes |
| Spearman | 0.747 | |
| c-index | 0.745 | |
| ranking rho | **0.646** | best of the models measured; baseline 0.619 |
| top-1 | **0.561** | baseline 0.491 |
| RMSE | 2.243 | **not comparable** — see below |

### Its five numbers are the best embedding-per-dimension here

A ridge on the van der Waals / hydrogen-bond / metal / hydrophobic / ionic terms reaches
**R 0.790 — 104% of what the model itself achieves** — while a 128-dimensional pooling of its
node embeddings reaches only 0.720.

Both halves are informative. The head scores *worse* than a linear probe on its own output
because it adds the terms **unweighted**: that is physics rather than regression, and fitting
the weights recovers what the constraint gives up. And five interpretable physical quantities
beating a learned 128-dimensional vector says the information sits in the bottleneck, which is
the whole claim of a physics-informed model — measured here rather than asserted.

For transfer this is the most interesting encoder in the roster: five dimensions carry nearly
everything, and each one means something.

**RMSE and MAE do not belong in a shared table for this model.** It predicts a binding free
energy in kcal/mol; the adapter negates it so the correlation runs the right way, but nothing
calibrates it to pK units. Correlation, c-index and ranking survive a linear map; error metrics
do not. GenScore has the same caveat for the same reason.

Best ranking power of anything measured so far is worth noting rather than burying: this is a
physics-informed model built for generalisation rather than for maximising a global
correlation, and the ranking numbers are where that shows.

## The dimorphite problem

`protonate.py` calls `dimorphite_dl.DimorphiteDL(...)`, and the current PyPI package no longer
provides that class — it was rewritten around a CLI-shaped API. Three shims were tried and all
failed: the importable entry points (`cli.run`, `cli.run_with_mol_list`) parse `sys.argv`, so
they see the caller's own flags, and hiding argv then trips a different check. The authors pin
no version, so this breaks on any fresh install rather than only for us.

The likely fix is vendoring the Durrant-lab original that has the class, rather than shimming
the rewrite.

## Hard-won facts (do NOT regress these)

- **`predict.py` imports modules that are not beside it.** `generate_data` and `protonate`
  live in `dataset/preprocess/`; the git history shows `generate_data.py` was moved out of
  `src/exe/` without the import following. PYTHONPATH covers it.
- **Python 3.9, which their README pins, no longer works**: dimorphite-dl publishes nothing
  for it. The image uses 3.10.
- **`Data.keys` became a method in torch_geometric 2.4.** `data.py` now accepts both
  spellings, so there is no upper bound on PyG any more. The same file also falls back from
  `torch_scatter.scatter` to `torch_geometric.utils.scatter` — note that call passes `dim=-1`
  explicitly, because the two disagree on the default and `energies_pairs` is
  (energy_types, pairs).
- **`aggr` must reach `MessagePassing.__init__`.** Setting `self.aggr` afterwards has been a
  no-op since PyG 2.1 and silently turns max-aggregation into sum. See above; this is the
  whole of the vdW divergence.
- **pymol is a real dependency**, not just the unused import at `predict.py:12` — protonate.py
  uses it. conda-forge ships it as pymol-open-source.
- Read the ligand from mol2 before sdf: several PDBbind SDFs do not sanitise and their
  `read_mols` returns `[None]`, which surfaces much later as `MolToSmiles(NoneType)`.

## Build & run

```bash
podman build --format=docker -t pignet2:latest .
gnnb run --variant pignet2.reference --capability predict --dataset <complexes>
```
