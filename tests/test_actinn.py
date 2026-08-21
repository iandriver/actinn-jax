"""End-to-end tests for the JAX ACTINN reference mapping."""

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import actinn_jax as ctp
from conftest import make_adata, make_hard_adata


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


def test_auto_schedule_frozen_at_and_above_the_pivot():
    """The published benchmark numbers were measured on references of 6,892 to
    584,944 cells with a fixed batch of 128-1024 and 50 epochs. Auto-scaling for
    small references must not touch anything in that range: at or above
    SMALL_REFERENCE_CELLS both rules have to reproduce the historical schedule
    exactly."""
    import actinn_jax.actinn_utils as au

    def historical_batch(n):                     # the pre-auto-epoch formula
        return int(min(1024, max(au.DEFAULT_BATCH_SIZE, n // 100)))

    benchmark_sizes = (6_892, 8_109, 12_800, 25_081, 46_911, 85_256, 584_944)
    for n in list(range(au.SMALL_REFERENCE_CELLS, 20_000, 97)) + list(benchmark_sizes):
        assert au.auto_batch_size(n) == historical_batch(n), n
        assert au.auto_num_epochs(n) == au.DEFAULT_NUM_EPOCHS, n

    assert au.auto_batch_size(12_800) == 128     # spot checks of the old test
    assert au.auto_batch_size(25_081) == 250     # scales with reference size
    assert au.auto_batch_size(46_911) == 469
    assert au.auto_batch_size(5_000_000) == 1024  # clamped at the top
    # every auto batch keeps a healthy number of gradient steps per epoch
    for n in (25_081, 46_911, 200_000):
        assert n // au.auto_batch_size(n) >= 100


def test_auto_schedule_bit_identical_above_the_pivot():
    """Not just the resolved hyper-parameters -- the trained weights themselves
    must be unchanged above the pivot, so cached references and published figures
    stay reproducible."""
    import actinn_jax.actinn_utils as au

    rng = np.random.default_rng(0)
    X = rng.random((5_200, 60)).astype(np.float32)
    Y = np.eye(5, dtype=np.float32)[rng.integers(0, 5, 5_200)]

    auto = au.train(X, Y, print_cost=False)
    historical = au.train(X, Y, num_epochs=50, batch_size=128, print_cost=False)
    assert all(np.array_equal(auto[k], historical[k]) for k in auto)


def test_small_reference_gets_enough_gradient_steps():
    """Below the pivot the schedule targets gradient steps rather than a fixed
    batch: 437 cells used to get ~170 updates in total, which is not training."""
    import actinn_jax.actinn_utils as au

    # the inherited schedule, for contrast
    assert 437 // au.DEFAULT_BATCH_SIZE * au.DEFAULT_NUM_EPOCHS < 200

    for n in (100, 437, 1_000, 2_000, 4_999):
        batch, epochs = au.auto_batch_size(n), au.auto_num_epochs(n)
        assert au.MIN_AUTO_BATCH_SIZE <= batch <= au.DEFAULT_BATCH_SIZE, n
        assert au.DEFAULT_NUM_EPOCHS <= epochs <= au.MAX_AUTO_NUM_EPOCHS, n
        steps = max(1, n // batch) * epochs
        assert steps >= au.MIN_USEFUL_TOTAL_STEPS, (n, steps)

    # continuous at the pivot: a 5000-cell reference maps to the fixed default,
    # so there is no jump in behaviour on either side of it.
    assert au.auto_num_epochs(5_000) == au.DEFAULT_NUM_EPOCHS
    assert au.auto_num_epochs(4_999) == au.DEFAULT_NUM_EPOCHS + 1


def test_small_reference_trains_instead_of_landing_near_chance():
    """The regression this guards: on a few-hundred-cell reference the inherited
    fixed schedule (50 epochs, batch 128) under-trains so badly that predictions
    are near chance -- silently, with no warning and no error. The default path
    has to actually fit."""
    train = make_hard_adata(n_cells=437, seed=1)
    query = make_hard_adata(n_cells=163, seed=2)
    chance = 1.0 / train.obs["celltype"].nunique()

    def accuracy(**train_kwargs):
        model = ctp.train_reference(train, train_label_name="celltype",
                                    print_cost=False, **train_kwargs)
        pred = model.predict_frame(query)[0].loc[query.obs_names, "celltype"].values
        return float((pred == query.obs["celltype"].values).mean())

    starved = accuracy(num_epochs=50, batch_size=128)   # what the defaults used to be
    default = accuracy()                                # what they are now

    assert starved < 5 * chance          # near chance, as reported
    assert default > 0.4                 # actually learns the classes
    assert default > 3 * starved         # and by a wide margin


def test_starved_training_warns():
    """A user who pins the hyper-parameters can still starve the fit; that must
    not be silent."""
    import actinn_jax.actinn_utils as au

    rng = np.random.default_rng(0)
    X = rng.random((400, 30)).astype(np.float32)
    Y = np.eye(4, dtype=np.float32)[rng.integers(0, 4, 400)]

    with pytest.warns(UserWarning, match="gradient steps"):
        au.train(X, Y, num_epochs=50, batch_size=128, print_cost=False)

    # the auto schedule on the same data clears the bar, so no warning fires
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        au.train(X, Y, print_cost=False)
