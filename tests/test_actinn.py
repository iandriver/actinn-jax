"""End-to-end tests for the JAX ACTINN reference mapping."""

import os

import numpy as np
import pandas as pd

import actinn_jax as ctp
from conftest import make_adata


def _accuracy(adata, pred_col="celltype_pred"):
    return float((adata.obs[pred_col].values == adata.obs["celltype"].values).mean())


def test_train_reference_and_predict_accuracy():
    train = make_adata(seed=1)
    query = make_adata(seed=2)  # same distribution, different cells

    model = ctp.train_reference(train, train_label_name="celltype", print_cost=False)
    adata, _ = ctp.predict(query, model, output_label_name="celltype_pred")

    assert "celltype_pred" in adata.obs
    assert "celltype_pred_probability" in adata.obs
    assert _accuracy(adata) > 0.85


def test_save_load_roundtrip(tmp_path):
    train = make_adata(seed=1)
    query = make_adata(seed=2)
    model = ctp.train_reference(train, print_cost=False)

    labels_before = model.predict_frame(query)[0]["celltype"].tolist()

    model.save(str(tmp_path / "ref"))
    reloaded = ctp.ReferenceModel.load(str(tmp_path / "ref"))
    labels_after = reloaded.predict_frame(query)[0]["celltype"].tolist()

    assert labels_before == labels_after


def test_query_gene_projection_robustness():
    """Shuffled gene order + missing/extra genes must still align correctly."""
    train = make_adata(seed=1)
    model = ctp.train_reference(train, print_cost=False)

    query = make_adata(seed=2)
    rng = np.random.default_rng(0)
    # Shuffle gene order, drop 50 genes, append novel genes absent from reference.
    order = rng.permutation(query.n_vars)
    query = query[:, order].copy()
    query = query[:, 50:].copy()
    extra_names = list(query.var_names) + [f"NOVEL{i}" for i in range(20)]
    import scipy.sparse as sp
    from anndata import AnnData
    X = sp.hstack([query.X, sp.csr_matrix((query.n_obs, 20))]).tocsr()
    query = AnnData(X=X, obs=query.obs.copy(),
                    var=pd.DataFrame(index=extra_names))

    adata, _ = ctp.predict(query, model, output_label_name="celltype_pred")
    assert _accuracy(adata) > 0.80


def test_celltype_predict_actinn_wrapper(tmp_path):
    train = make_adata(seed=1)
    train.write_h5ad(str(tmp_path / "train.h5ad"))
    query = make_adata(seed=2)
    original_order = list(query.obs_names)

    adata, params = ctp.celltype_predict_actinn(
        query,
        str(tmp_path / "train.h5ad"),
        str(tmp_path),
        train_label_name="celltype",
        output_label_name="celltype_pred",
    )

    # obs order preserved, columns added, output files written.
    assert list(adata.obs_names) == original_order
    assert _accuracy(adata) > 0.85
    assert os.path.exists(tmp_path / "celltype_pred_predicted_label.txt")
    assert os.path.exists(tmp_path / "train_predicted_probabilities.txt")
    assert "W1" in params
