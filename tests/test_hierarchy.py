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


def test_refine_to_query_drops_absent_keeps_present():
    """refine_to_query on a query missing most of the reference's types should keep
    every type genuinely present (no false negatives) and drop unevidenced ones."""
    train = make_adata(n_types=8, n_per_type=60, seed=1)
    query_full = make_adata(n_types=8, n_per_type=40, seed=2)
    present = {"type_0", "type_1", "type_2"}
    query = query_full[query_full.obs["celltype"].isin(present)].copy()
    emb = _embeddings_from_labels(train, dim=16)

    model = aj.build_hierarchical_reference(train, "celltype", emb, n_groups=4,
                                            print_cost=False)
    refined = aj.refine_to_query(model, query)
    kept = set().union(*refined.allowed_classes.values()) if refined.allowed_classes else set()

    assert present <= kept, f"dropped a present type: {present - kept}"
    absent = set(model.classes) - present
    assert not (absent & kept), f"kept an absent type: {absent & kept}"

    out = refined.predict(query.copy(), output_label_name="pred")
    acc = (out.obs["pred"].values == out.obs["celltype"].values).mean()
    assert acc > 0.85
    # renormalized probabilities are still valid probabilities
    assert (out.obs["pred_probability"] >= 0).all() and (out.obs["pred_probability"] <= 1).all()


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


# --------------------------------------------------------------------------- #
# Tissue-aware refinement
# --------------------------------------------------------------------------- #
def _tissue_model(seed=1):
    """A small hierarchical model with a hand-set class_to_tissue map."""
    train = make_adata(n_types=6, seed=seed)
    emb = _embeddings_from_labels(train)
    model = aj.build_hierarchical_reference(train, "celltype", emb, n_groups=3,
                                            print_cost=False)
    t = sorted(model.classes)             # type_0 .. type_5
    model.class_to_tissue = {
        t[0]: ["liver"], t[1]: ["lung"], t[2]: ["*"], t[3]: ["liver", "lung"],
        # t[4], t[5] intentionally unmapped -> always allowed
    }
    return model, t


def _allowed_names(refined):
    return set().union(*refined.allowed_classes.values()) if refined.allowed_classes else set()


def test_refine_to_tissue_prunes_wrong_tissue():
    model, t = _tissue_model()
    allowed = _allowed_names(aj.refine_to_tissue(model, tissue="liver"))
    assert t[0] in allowed            # liver-specific kept
    assert t[1] not in allowed        # lung-specific pruned
    assert t[2] in allowed            # pan-tissue ("*") kept
    assert t[3] in allowed            # multi-tissue incl. liver kept
    assert t[4] in allowed and t[5] in allowed   # unmapped -> always allowed


def test_refine_to_tissue_synonym_and_predict():
    train = make_adata(n_types=6, seed=1)
    emb = _embeddings_from_labels(train)
    model = aj.build_hierarchical_reference(train, "celltype", emb, n_groups=3,
                                            print_cost=False)
    t = sorted(model.classes)
    model.class_to_tissue = {t[0]: ["liver"], t[1]: ["lung"], t[2]: ["*"],
                             t[3]: ["blood"], t[4]: ["liver"], t[5]: ["blood"]}
    query = make_adata(n_types=6, seed=2)
    # 'PBMC' -> blood; only pan-tissue and blood-specific types survive.
    out = aj.refine_to_tissue(model, tissue="PBMC").predict(query, output_label_name="p")
    assert set(out.obs["p"]).issubset({t[2], t[3], t[5]})


def test_refine_to_tissue_auto_from_obs():
    model, t = _tissue_model()
    query = make_adata(n_types=6, seed=2)
    query.obs["tissue"] = "liver"
    allowed = _allowed_names(aj.refine_to_tissue(model, adata=query))   # tissue=None -> auto
    assert t[0] in allowed and t[1] not in allowed


def test_refine_to_query_composes_with_tissue():
    model, t = _tissue_model()
    query = make_adata(n_types=6, seed=2)
    refined = aj.refine_to_query(model, query, tissue="liver")
    allowed = _allowed_names(refined)
    assert t[1] not in allowed        # lung-specific never survives a liver filter


def test_tissue_map_roundtrips(tmp_path):
    model, _ = _tissue_model()
    model.save(str(tmp_path / "h"))
    reloaded = aj.HierarchicalReferenceModel.load(str(tmp_path / "h"))
    assert reloaded.class_to_tissue == model.class_to_tissue


def test_refine_to_tissue_noop_without_map():
    """A model with no tissue map imposes no filter (allowed_* stay None)."""
    train = make_adata(n_types=4, seed=1)
    emb = _embeddings_from_labels(train)
    model = aj.build_hierarchical_reference(train, "celltype", emb, n_groups=2,
                                            print_cost=False)
    refined = aj.refine_to_tissue(model, tissue="liver")
    assert refined.allowed_groups is None and refined.allowed_classes is None
