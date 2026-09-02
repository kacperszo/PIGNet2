# PIGNet2, CPU reference tier.
#
# Their requirements.txt pins torch 1.9.1+cu111, which predates our GPUs; CPU is the right
# tier here anyway. Two things the file does not tell you:
#
#   * `predict.py` imports `generate_data` and `protonate` flat, but both live in
#     dataset/preprocess/ — the git history shows generate_data.py was moved out of src/exe/
#     without the import following it. PYTHONPATH covers the gap.
#   * pymol is a real dependency, not just the unused import at predict.py:12. protonate.py
#     uses it. conda-forge ships it as pymol-open-source.
FROM docker.io/mambaorg/micromamba:1.5.8

USER root
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

RUN micromamba install -y -n base -c conda-forge -c bioconda \
        # 3.10, not the 3.9 their README pins: dimorphite-dl now publishes nothing for 3.9,
        # so the authors' own install line no longer resolves
        python=3.10 pymol-open-source openbabel libstdcxx-ng \
        # protonate_pdb shells out to `reduce` to add hydrogens to the protein; without it
        # every complex dies on FileNotFoundError deep inside preprocessing. It lives in
        # bioconda, not conda-forge.
        reduce \
    && micromamba clean --all --yes
ENV PATH=/opt/conda/bin:${PATH}
# conda's libstdc++ ahead of the system one, as for GenScore
ENV LD_LIBRARY_PATH=/opt/conda/lib

RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
# torch_geometric pinned below 2.4: that release turned `Data.keys` from a property into a
# method, and their data.py does `set(ligand.keys)`, which then raises
# "'method' object is not iterable" from inside complex_to_data.
RUN pip install --no-cache-dir "torch_geometric==2.3.1" \
    && pip install --no-cache-dir torch_scatter torch_sparse \
        -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

RUN pip install --no-cache-dir \
    # rdkit pinned to the era of the committed golden file (examples/case1.txt, 2024-03).
    # Node features depend on RDKit's bond and hybridisation perception, and the Morse vdW
    # term is the only energy that reads them — the other four use table radii and scalar
    # coefficients, which is why they matched a 2025 RDKit exactly while vdW did not.
    "rdkit==2023.9.6" \
    biopython \
    "hydra-core>=1.2" \
    "omegaconf>=2.2" \
    "numpy<2" \
    scipy \
    pandas \
    scikit-learn \
    tqdm \
    # pinned: the authors leave it unversioned, and current releases dropped the
    # DimorphiteDL class their protonate.py calls. 1.2.5 is the last with that API and
    # the first that supports python 3.10.
    "dimorphite-dl==1.2.5"

# the flat imports predict.py expects
ENV PYTHONPATH=/work:/work/src:/work/src/exe:/work/dataset/preprocess

RUN python -c "import torch, torch_geometric, torch_scatter, rdkit, pymol, Bio, omegaconf; \
print('env ok', torch.__version__)"

WORKDIR /work
COPY . /work
RUN cd /work/src/exe && python -c "import generate_data, protonate; print('pignet2 imports ok')"
