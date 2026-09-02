"""Import-time stand-in for dimorphite_dl in the authors-pinned image.

`protonate.py` imports this at module level, so it is needed to import at all — but the
reference image runs with protonation off, so nothing here is ever called. Using a stub keeps
the image on Python 3.9, which torch 1.9.1 requires and which dimorphite-dl no longer
publishes for.

Anything that does call it fails loudly rather than silently returning the input, because a
silently unprotonated ligand would change the hydrogen-bond term without saying so.
"""


class DimorphiteDL:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "dimorphite_dl is a stub in this image: protonation is disabled here on purpose. "
            "Use the modern tier if you need it."
        )
