"""Novel / unknown cell-type discovery.

A genuinely novel cell type — one absent from the reference — shows up not as a few
scattered low-confidence cells but as a **coherent group the reference cannot
confidently explain**: a cluster whose cells are consistently forced to a label with
low probability. :func:`detect_novel_celltypes` looks for exactly that — clusters
that are (a) large enough to be a real population and (b) predominantly low
confidence — so novel types get flagged rather than silently mislabeled.

This is cluster-level rejection, deliberately distinct from the per-cell ``min_prob``
abstain in :meth:`HierarchicalReferenceModel.predict`: abstain marks individual
uncertain cells; this finds uncertain *populations*.
"""
import warnings

import numpy as np
import pandas as pd

from .actinn_predict import _extract, _normalize


def _quick_cluster(adata, resolution, use_raw):
    """Leiden clustering on the query (used when no ``cluster_key`` is supplied)."""
    import anndata as ad
    import scanpy as sc

    X, _, _ = _extract(adata, use_raw=use_raw)
    tmp = ad.AnnData(_normalize(X))                       # cp10k + log2, sparse
    sc.pp.pca(tmp, n_comps=int(min(30, tmp.n_vars - 1, tmp.n_obs - 1)))
    sc.pp.neighbors(tmp, n_neighbors=15)
    try:                                                  # fast igraph flavor if available
        sc.tl.leiden(tmp, resolution=resolution, flavor="igraph",
                     n_iterations=2, directed=False)
    except (ImportError, TypeError, ValueError):
        try:
            sc.tl.leiden(tmp, resolution=resolution)
        except ImportError as e:                          # no leiden backend installed
            raise ImportError(
                "Automatic clustering needs 'leidenalg' (or igraph). Install it, or "
                "pass cluster_key= pointing at an existing clustering in adata.obs."
            ) from e
    return tmp.obs["leiden"].to_numpy()


def _novel_markers(adata, clusters, novel_map, use_raw, n_markers):
    """Top up-regulated genes for each novel cluster (mean log-norm expression in the
    cluster minus the rest) — a quick handle for characterizing what the type is."""
    X, genes, _ = _extract(adata, use_raw=use_raw)
    Xn = _normalize(X)
    genes = np.asarray(genes, dtype=object)
    out = {}
    for cl, name in novel_map.items():
        m = clusters == cl
        mu_in = np.asarray(Xn[m].mean(axis=0)).ravel()
        mu_out = np.asarray(Xn[~m].mean(axis=0)).ravel()
        top = np.argsort(mu_in - mu_out)[::-1][:n_markers]
        out[name] = list(genes[top])
    return out


def detect_novel_celltypes(model, adata, cluster_key=None, min_prob=0.5,
                           min_cells=10, min_frac=0.5, use_raw="auto",
                           chunk_size=50000, resolution=1.0, n_markers=10,
                           output_key="novel"):
    """Flag coherent groups of cells the reference cannot confidently explain.

    Clusters the query (or uses an existing clustering), scores each cell's best
    reference probability, and marks a cluster as a **candidate novel cell type**
    when it is both large enough and mostly low-confidence.

        evidence, markers = aj.detect_novel_celltypes(model, adata, min_cells=10)
        evidence[evidence.novel]                 # candidate novel populations
        markers['novel_1']                       # its top marker genes
        adata.obs['novel']                       # 'novel_1'… on flagged cells, '' otherwise

    Parameters
    ----------
    cluster_key : obs column of an existing clustering. If ``None``, a Leiden
        clustering is computed (needs ``leidenalg``/igraph).
    min_prob : a cell is "low confidence" when its best label probability < this.
    min_cells : **minimum cells for a cluster to be called novel** — the size floor
        that separates a real novel population from noise/doublets. Lower it (e.g.
        ``min_cells=5``) to surface rarer candidates at the cost of more false alarms.
    min_frac : fraction of a cluster that must be low-confidence to flag it.
    n_markers : number of top marker genes returned per novel cluster.
    output_key : ``adata.obs`` column to write novel labels into (plus
        ``<output_key>_confidence`` with each cell's best probability).

    Returns
    -------
    evidence : DataFrame, one row per cluster (``n_cells``, ``frac_low_conf``,
        ``median_conf``, ``nearest_label`` = what the reference would have called it,
        ``novel`` = flag), most-novel first.
    markers : ``{novel_label: [gene, ...]}`` for the flagged clusters.
    """
    frame, _ = model.predict_frame(adata, use_raw=use_raw, chunk_size=chunk_size)
    conf = frame["celltype_probability"].to_numpy()
    pred = frame["celltype"].to_numpy().astype(object)
    low = conf < min_prob

    if cluster_key is not None:
        if cluster_key not in adata.obs:
            raise KeyError(f"cluster_key {cluster_key!r} not in adata.obs")
        clusters = adata.obs[cluster_key].astype(str).to_numpy()
    else:
        clusters = _quick_cluster(adata, resolution, use_raw)

    rows = []
    for c in pd.unique(clusters):
        m = clusters == c
        n = int(m.sum())
        frac = float(low[m].mean())
        vals, counts = np.unique(pred[m], return_counts=True)
        rows.append({"cluster": str(c), "n_cells": n,
                     "frac_low_conf": round(frac, 3),
                     "median_conf": round(float(np.median(conf[m])), 3),
                     "nearest_label": vals[counts.argmax()],
                     "novel": bool(n >= min_cells and frac >= min_frac)})
    evidence = (pd.DataFrame(rows)
                .sort_values(["novel", "frac_low_conf", "n_cells"],
                             ascending=[False, False, False])
                .reset_index(drop=True))

    novel_map = {r["cluster"]: f"novel_{i + 1}"
                 for i, r in enumerate(evidence[evidence.novel].to_dict("records"))}
    labels = np.array([novel_map.get(str(c), "") for c in clusters], dtype=object)
    adata.obs[output_key] = pd.Series(labels, index=adata.obs_names)
    adata.obs[output_key + "_confidence"] = pd.Series(conf, index=adata.obs_names)

    if not novel_map:
        warnings.warn(
            f"actinn-jax: no novel cell-type candidates (no cluster had >= {min_cells} "
            f"cells with >= {min_frac:.0%} low-confidence). Lower min_cells/min_prob to "
            "search harder.", stacklevel=2)
    markers = _novel_markers(adata, clusters, novel_map, use_raw, n_markers) if novel_map else {}
    return evidence, markers
