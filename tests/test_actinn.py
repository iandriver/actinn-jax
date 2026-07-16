"""End-to-end tests for the JAX ACTINN reference mapping."""

import os

import numpy as np
import pandas as pd
import pytest

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


def test_gene_id_mismatch_raises_clear_error():
    """A query whose gene ids don't match the reference (and has no usable .var
    column to fall back to) must fail loudly, not silently emit one constant
    label (the Ensembl-vs-symbol footgun)."""
    train = make_adata(seed=1)
    query = make_adata(seed=2)
    # Rename every query gene so nothing overlaps the reference's gene set.
    query.var_names = ["MISMATCH_" + g for g in query.var_names]

    model = ctp.train_reference(train, train_label_name="celltype", print_cost=False)
    with pytest.raises(ValueError, match="gene-identifier mismatch"):
        ctp.predict(query, model, output_label_name="celltype_pred")


def test_auto_matches_gene_ids_from_var_column():
    """If var_names don't match but a .var column does, prediction should fall
    back to that column automatically (by default) and annotate correctly."""
    train = make_adata(seed=1)
    query = make_adata(seed=2)
    # Stash the real (matching) ids in a column, then scramble var_names.
    query.var["ensembl_id"] = list(query.var_names)
    query.var_names = ["SYMBOL_%d" % i for i in range(query.n_vars)]

    model = ctp.train_reference(train, train_label_name="celltype", print_cost=False)
    with pytest.warns(UserWarning, match="ensembl_id"):
        adata, _ = ctp.predict(query, model, output_label_name="celltype_pred")
    # Auto-matching recovers the signal, so accuracy matches the aligned case.
    assert _accuracy(adata) > 0.85


def test_matching_var_names_are_left_untouched():
    """When var_names already match the reference, no fallback/warning fires."""
    import warnings

    train = make_adata(seed=1)
    query = make_adata(seed=2)
    query.var["ensembl_id"] = ["DECOY_%d" % i for i in range(query.n_vars)]

    model = ctp.train_reference(train, train_label_name="celltype", print_cost=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any spurious remap warning would raise
        adata, _ = ctp.predict(query, model, output_label_name="celltype_pred")
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


def test_standardize_stores_scaler_and_roundtrips(tmp_path):
    train = make_adata(seed=1)
    query = make_adata(seed=2)

    base = ctp.train_reference(train, train_label_name="celltype",
                              standardize=False, print_cost=False)
    std = ctp.train_reference(train, train_label_name="celltype",
                              standardize=True, print_cost=False)

    # opt-in stores a frozen scaler; default leaves it off
    assert base.mu is None and base.sd is None
    assert std.mu is not None and std.sd is not None
    assert std.mu.shape == std.sd.shape == std.select_idx.shape
    assert np.all(std.sd > 0)  # zero std guarded to 1.0

    # standardization changes predictions but still classifies the query well
    a_std, _ = ctp.predict(query, std, output_label_name="celltype_pred")
    assert _accuracy(a_std) > 0.85

    # scaler survives save/load: identical labels before/after
    before = std.predict_frame(query)[0]["celltype"].tolist()
    std.save(str(tmp_path / "std"))
    reloaded = ctp.ReferenceModel.load(str(tmp_path / "std"))
    assert reloaded.mu is not None and reloaded.mu.shape == std.mu.shape
    assert reloaded.predict_frame(query)[0]["celltype"].tolist() == before


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
