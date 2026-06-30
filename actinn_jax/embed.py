"""Optional foundation-model embedding of a reference, for building hierarchies.

scPRINT (PyTorch + lamindb, GPU-oriented) is **not** a dependency of actinn-jax — it is
imported lazily here and only needed to *build* a new hierarchical reference. Annotating
data never needs it. Install separately:

    pip install scprint            # + lamindb bionty populated for human AND mouse
    # download a checkpoint, e.g. medium-v1.5.ckpt from huggingface.co/jkobject/scPRINT

Returns a ``(n_cells, 256)`` embedding aligned to ``adata`` row order, suitable for
``actinn_jax.build_hierarchical_reference(ref, label, embeddings)``.
"""
import os

import numpy as np

_SCHEMA = ("cell_type_ontology_term_id", "assay_ontology_term_id",
           "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id",
           "sex_ontology_term_id", "development_stage_ontology_term_id",
           "tissue_ontology_term_id")


def _patch_mps_autocast():
    """torch.autocast('mps') is unsupported; make it a no-op so scPRINT runs on MPS."""
    import contextlib
    import torch
    if getattr(torch.autocast, "_actinn_patched", False):
        return
    orig = torch.autocast

    class _AC:
        _actinn_patched = True

        def __init__(self, device_type, **kw):
            self._cm = (orig(device_type, **kw) if device_type in ("cuda", "cpu", "xpu")
                        else contextlib.nullcontext())

        def __enter__(self):
            return self._cm.__enter__()

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

    torch.autocast = _AC


def scprint_embed(adata, ckpt="medium-v1.5.ckpt", device="auto",
                  organism="NCBITaxon:9606", batch_size=32, max_len=2000):
    """Embed ``adata`` (raw counts, Ensembl var_names) with scPRINT. Returns (n, 256)."""
    import anndata as ad
    import scipy.sparse as sp
    import torch
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _patch_mps_autocast()
    from scprint import scPrint
    from scprint.tasks import Embedder
    from scdataloader import Preprocessor
    try:
        from scdataloader.preprocess import additional_preprocess
    except Exception:
        additional_preprocess = None

    X = adata.raw.X if adata.raw is not None else adata.X
    var = adata.raw.var.copy() if adata.raw is not None else adata.var.copy()
    a = ad.AnnData(X=sp.csr_matrix(X).astype(np.float32), obs=adata.obs[[]].copy(), var=var)
    a.obs["organism_ontology_term_id"] = organism
    for c in _SCHEMA:
        a.obs[c] = "unknown"
    a.obs["suspension_type"] = "cell"
    a.obs["is_primary_data"] = True

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = scPrint.load_from_checkpoint(ckpt, precpt_gene_emb=None).to(device)

    pp = Preprocessor(do_postp=False, force_preprocess=True,
                      **({"additional_preprocess": additional_preprocess} if additional_preprocess else {}))
    a = pp(a)
    emb = Embedder(doclass=False, precision="32", dtype=torch.float32,
                   batch_size=batch_size, num_workers=0, doplot=False, max_len=max_len)
    res = emb(model, a)
    out = res[0] if isinstance(res, (tuple, list)) else res
    assert out.n_obs == adata.n_obs, f"cell count changed {adata.n_obs}->{out.n_obs}"
    return np.asarray(out.obsm["scprint_emb"])
