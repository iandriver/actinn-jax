"""Synthetic single-cell data generator shared by the tests."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def make_adata(n_per_type=80, n_genes=1500, n_types=4, seed=0, gene_prefix="G"):
    """Build a synthetic raw-count AnnData with linearly separable cell types.

    Each type over-expresses a disjoint block of marker genes, so a small model
    can learn it. Returns an AnnData with sparse counts in ``X`` and a
    ``celltype`` column in ``obs``.
    """
    rng = np.random.default_rng(seed)
    block = n_genes // (n_types + 1)
    rows = []
    labels = []
    for t in range(n_types):
        base = rng.poisson(0.2, size=(n_per_type, n_genes)).astype(np.float32)
        lo, hi = t * block, (t + 1) * block
        base[:, lo:hi] += rng.poisson(8.0, size=(n_per_type, block)).astype(np.float32)
        rows.append(base)
        labels.extend([f"type_{t}"] * n_per_type)

    X = np.vstack(rows)
    perm = rng.permutation(X.shape[0])
    X, labels = X[perm], list(np.array(labels)[perm])

    var = pd.DataFrame(index=[f"{gene_prefix}{i}" for i in range(n_genes)])
    obs = pd.DataFrame(
        {"celltype": labels},
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    return AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
