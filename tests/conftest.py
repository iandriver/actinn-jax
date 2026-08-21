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


def make_hard_adata(n_cells=437, n_types=31, n_genes=1000, seed=0):
    """A small, *hard* reference in the shape of the Open Problems
    ``label_projection`` smoke-test resource: a few hundred cells spread over
    ~30 imbalanced classes, with overlapping 25-gene marker programs and
    per-class library-size drift.

    Unlike :func:`make_adata` the classes are not trivially separable, so the
    number of gradient steps the training schedule actually takes shows up in
    held-out accuracy. Reference and query built with different ``seed`` values
    are drawn from the same distribution.
    """
    # One fixed draw defines the classes themselves, so every dataset built here
    # shares the same marker programs regardless of ``seed``.
    fixed = np.random.default_rng(7)
    programs = [fixed.choice(n_genes, 25, replace=False) for _ in range(n_types)]
    depth = fixed.uniform(0.5, 2.0, n_types)
    freq = fixed.dirichlet(np.full(n_types, 2.0))

    rng = np.random.default_rng(seed)
    y = rng.choice(n_types, n_cells, p=freq)
    lam = np.full((n_cells, n_genes), 0.3)
    for t in range(n_types):
        rows = y == t
        lam[np.ix_(rows, programs[t])] += 1.6
        lam[rows] *= depth[t]

    X = sp.csr_matrix(rng.poisson(lam).astype(np.float32))
    obs = pd.DataFrame({"celltype": [f"type_{i}" for i in y]},
                       index=[f"cell_{seed}_{i}" for i in range(n_cells)])
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
