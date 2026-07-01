"""Tests for the two-stage hierarchical workflow."""

import numpy as np

import actinn_jax as aj
from conftest import make_adata


def _embeddings_from_labels(adata, dim=16, seed=0):
    """Toy embedding that separates cell types (stands in for scPRINT vectors)."""
    rng = np.random.default_rng(seed)
    types = list(adata.obs["celltype"].unique())
    centers = {t: rng.normal(size=dim) * 5 for t in types}
    return np.vstack([centers[t] + rng.normal(scale=0.3, size=dim)
                      for t in adata.obs["celltype"]])


def test_discover_hierarchy_groups_types():
    train = make_adata(n_types=6, seed=1)
    emb = _embeddings_from_labels(train)
    grp = aj.discover_hierarchy(emb, train.obs["celltype"].to_numpy(), n_groups=3)
    assert set(grp) == set(train.obs["celltype"].unique())
    assert 1 <= len(set(grp.values())) <= 3


def test_build_annotate_roundtrip(tmp_path):
    train = make_adata(n_types=6, seed=1)
    query = make_adata(n_types=6, seed=2)
    emb = _embeddings_from_labels(train)

    model = aj.build_hierarchical_reference(train, "celltype", emb, n_groups=3,
                                            print_cost=False)
    out = aj.annotate(query, model, output_label_name="pred")
    acc = (out.obs["pred"].values == out.obs["celltype"].values).mean()
    assert acc > 0.85
    assert "pred_coarse" in out.obs

    # save / load roundtrip yields identical predictions
    model.save(str(tmp_path / "hier"))
    reloaded = aj.HierarchicalReferenceModel.load(str(tmp_path / "hier"))
    f1 = model.predict_frame(query)[0]["celltype"].tolist()
    f2 = reloaded.predict_frame(query)[0]["celltype"].tolist()
    assert f1 == f2


def test_build_from_precomputed_hierarchy():
    """Build from a {type: group} dict (embedding cells needn't align to ref);
    a label missing from the hierarchy falls into a catch-all group."""
    train = make_adata(n_types=6, seed=1)
    query = make_adata(n_types=6, seed=2)
    types = sorted(train.obs["celltype"].unique())
    hier = {t: str(i % 2) for i, t in enumerate(types)}
    hier.pop(types[0])                       # drop one -> exercises the fallback group
    model = aj.build_hierarchical_reference(train, "celltype", hierarchy=hier,
                                            print_cost=False)
    assert model.type_to_group[types[0]] == "_unmapped"
    out = aj.annotate(query, model, output_label_name="pred")
    assert (out.obs["pred"].values == out.obs["celltype"].values).mean() > 0.7


def test_abstain_threshold():
    """min_prob relabels low-confidence calls as 'unknown' (OOD abstain)."""
    train = make_adata(n_types=6, seed=1)
    query = make_adata(n_types=6, seed=2)
    emb = _embeddings_from_labels(train)
    model = aj.build_hierarchical_reference(train, "celltype", emb, n_groups=3,
                                            print_cost=False)
    # softmax max prob is always < 1.0, so min_prob=1.0 abstains on every cell
    frame = model.predict_frame(query, min_prob=1.0)[0]
    assert (frame["celltype"] == "unknown").all()
    # with no threshold, nothing is forced to unknown by abstain
    frame0 = model.predict_frame(query, min_prob=0.0)[0]
    assert (frame0["celltype"] == "unknown").mean() < 0.5
