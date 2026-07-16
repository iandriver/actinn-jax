"""Tests for novel / unknown cell-type discovery (cluster-level rejection)."""

import numpy as np
import pytest

import actinn_jax as aj
from conftest import make_adata


def _reference_without(label):
    """A flat reference trained on all synthetic types except ``label``."""
    train = make_adata(n_types=6, seed=1)
    train = train[train.obs["celltype"] != label].copy()
    return aj.train_reference(train, train_label_name="celltype", print_cost=False)


def test_flags_held_out_type_as_novel():
    """A cell type withheld from the reference is flagged; known ones are not."""
    model = _reference_without("type_5")
    query = make_adata(n_types=6, seed=2)          # contains type_5
    evidence, markers = aj.detect_novel_celltypes(
        model, query, cluster_key="celltype", min_prob=0.75, min_cells=20)
    flagged = set(evidence.loc[evidence.novel, "cluster"])
    assert "type_5" in flagged                     # the unseen type is caught
    assert flagged == {"type_5"}                   # and nothing known is flagged
    assert (query.obs["novel"] != "").sum() == (query.obs["celltype"] == "type_5").sum()
    assert markers and all(len(v) for v in markers.values())


def test_min_cells_floor_is_tunable():
    """Raising min_cells above the population size stops it being called novel."""
    model = _reference_without("type_5")
    query = make_adata(n_types=6, seed=2)
    n5 = int((query.obs["celltype"] == "type_5").sum())
    lo, _ = aj.detect_novel_celltypes(model, query, cluster_key="celltype", min_prob=0.75, min_cells=n5)
    assert "type_5" in set(lo.loc[lo.novel, "cluster"])
    with pytest.warns(UserWarning, match="no novel cell-type candidates"):
        hi, _ = aj.detect_novel_celltypes(
            model, query, cluster_key="celltype", min_prob=0.75, min_cells=n5 + 1)
    assert not hi["novel"].any()


def test_writes_obs_and_confidence():
    model = _reference_without("type_5")
    query = make_adata(n_types=6, seed=2)
    aj.detect_novel_celltypes(model, query, cluster_key="celltype", min_prob=0.75,
                              min_cells=20, output_key="nov")
    assert "nov" in query.obs and "nov_confidence" in query.obs
    assert query.obs["nov"].str.startswith("novel_").any()
