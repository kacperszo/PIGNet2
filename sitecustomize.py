"""Restore `dimorphite_dl.DimorphiteDL`, which the current release no longer provides.

`dataset/preprocess/protonate.py` calls `dimorphite_dl.DimorphiteDL(**params).protonate(smi)`.
The package on PyPI is the same Durrant-lab code, but it was repackaged around a CLI: the
class is gone, and the importable entry points (`cli.run`, `cli.run_with_mol_list`) run their
own argument parser, so they see the calling script's flags and abort. The authors pin no
version, so their own `predict.py` fails on a fresh install too — verified by running their
README example, which dies at the same line.

`mol.Protonate` is the actual engine and takes a plain dict, no argparse involved. This wraps
it back into the class shape their code expects, so their pipeline runs unmodified and
protonation stays on — which matters here, because this model computes hydrogen-bond and ionic
terms explicitly and was trained on protonated input.

Named sitecustomize.py so Python loads it automatically, before any of their imports.
"""

try:
    import dimorphite_dl

    if not hasattr(dimorphite_dl, "DimorphiteDL"):
        from dimorphite_dl.mol import Protonate as _Protonate

        class _DimorphiteDL:
            def __init__(self, min_ph=6.4, max_ph=8.4, pka_precision=1.0, max_variants=128):
                self._args = {
                    "min_ph": min_ph,
                    "max_ph": max_ph,
                    "pka_precision": pka_precision,
                    "max_variants": max_variants,
                    "silent": True,
                }

            def protonate(self, smiles):
                # Protonate mutates the dict it is given, so hand it a fresh copy each call
                out = list(_Protonate({**self._args, "smiles": smiles}))
                # entries come back as "<smiles>\t<name>"
                return [line.split("\t")[0].strip() for line in out] or [smiles]

        dimorphite_dl.DimorphiteDL = _DimorphiteDL
except ImportError:
    pass
