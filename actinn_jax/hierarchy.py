"""Two-stage (coarse -> fine) hierarchical annotation for actinn-jax.

A foundation-model embedding of the *reference* (e.g. scPRINT; see ``actinn_jax.embed``)
is used **once, offline** to discover a coarse->fine cell-type hierarchy. We then train a
coarse classifier plus one fine classifier per coarse group. Annotating new data needs
**only actinn-jax on CPU** — no embedding, no GPU at inference.

    from actinn_jax import build_hierarchical_reference, annotate
    model = build_hierarchical_reference(ref_adata, "cell_type", embeddings)  # offline
    model.save("my_reference")
    ...
    model = HierarchicalReferenceModel.load("my_reference")
    adata = annotate(query_adata, model)            # -> obs['celltype'] (+ _coarse, _probability)
"""
import json
import os

import numpy as np
import pandas as pd

from .actinn_predict import ReferenceModel, train_reference, _load_adata


def discover_hierarchy(embeddings, labels, n_groups=8):
    """Cluster cell-type centroids in embedding space into ``n_groups`` coarse groups.

    Parameters
    ----------
    embeddings : array (n_cells, n_dim)
        Per-cell embedding of the reference (e.g. scPRINT 256-d vectors).
    labels : array (n_cells,)
        Fine cell-type labels.

    Returns
    -------
    dict {cell_type: group_id(str)}
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    labels = np.asarray(labels).astype(str)
    types = np.unique(labels)
    if len(types) <= n_groups:
        return {t: str(i) for i, t in enumerate(types)}
    centroids = np.vstack([np.asarray(embeddings)[labels == t].mean(0) for t in types])
    groups = fcluster(linkage(centroids, "ward"), n_groups, criterion="maxclust")
    return {t: str(int(g)) for t, g in zip(types, groups)}


class HierarchicalReferenceModel:
    """A coarse ReferenceModel + one fine ReferenceModel per coarse group."""

    def __init__(self, coarse, fine, type_to_group, classes, class_to_cl=None):
        self.coarse = coarse                  # ReferenceModel over coarse groups
        self.fine = fine                      # {group: ReferenceModel | single-label str}
        self.type_to_group = dict(type_to_group)
        self.classes = list(classes)
        # optional {cell_type_name: Cell-Ontology id} for ontology-aware roll-up/eval
        self.class_to_cl = dict(class_to_cl) if class_to_cl else {}

    def predict_frame(self, adata, use_raw="auto", chunk_size=50000, min_prob=None):
        """Coarse-predict, route each cell to its group's fine model, return a frame.

        ``min_prob`` (0-1, optional): cells whose final fine-label probability is below
        this are relabeled ``"unknown"`` (abstain), so out-of-distribution cells — types
        not in the reference — are flagged rather than force-labeled.
        """
        cframe, _ = self.coarse.predict_frame(adata, use_raw=use_raw, chunk_size=chunk_size)
        coarse_pred = cframe["celltype"].to_numpy()
        coarse_prob = cframe["celltype_probability"].to_numpy()
        out = np.empty(adata.n_obs, dtype=object)
        prob = np.zeros(adata.n_obs, dtype=np.float32)
        for g in np.unique(coarse_pred):
            mask = coarse_pred == g
            fm = self.fine.get(str(g))
            if fm is None:
                out[mask] = "unknown"
            elif isinstance(fm, str):           # group has a single fine type
                out[mask] = fm
                prob[mask] = coarse_prob[mask]
            else:
                ff, _ = fm.predict_frame(adata[mask], use_raw=use_raw, chunk_size=chunk_size)
                out[mask] = ff["celltype"].to_numpy()
                prob[mask] = ff["celltype_probability"].to_numpy()
        if min_prob is not None:
            out[prob < min_prob] = "unknown"
        return pd.DataFrame({"celltype": out, "celltype_probability": prob,
                             "coarse": coarse_pred}, index=list(adata.obs_names)), None

    def predict(self, adata, output_label_name="celltype", use_raw="auto",
                chunk_size=50000, min_prob=None):
        """Annotate ``adata`` in place: adds ``celltype``, ``_probability``, ``_coarse``."""
        frame, _ = self.predict_frame(adata, use_raw=use_raw, chunk_size=chunk_size,
                                      min_prob=min_prob)
        adata.obs[output_label_name] = frame.loc[adata.obs.index, "celltype"]
        adata.obs[output_label_name + "_probability"] = frame.loc[adata.obs.index, "celltype_probability"]
        adata.obs[output_label_name + "_coarse"] = frame.loc[adata.obs.index, "coarse"]
        return adata

    # -- persistence: a directory of ReferenceModel files + a manifest ------- #
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.coarse.save(os.path.join(path, "coarse"))
        fine_manifest = {}
        for g, fm in self.fine.items():
            if isinstance(fm, str):
                fine_manifest[g] = {"single_label": fm}
            else:
                fm.save(os.path.join(path, f"fine_{g}"))
                fine_manifest[g] = {"single_label": None}
        with open(os.path.join(path, "manifest.json"), "w") as fh:
            json.dump({"type_to_group": self.type_to_group, "classes": self.classes,
                       "class_to_cl": self.class_to_cl, "fine": fine_manifest}, fh)
        return path

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "manifest.json")) as fh:
            man = json.load(fh)
        coarse = ReferenceModel.load(os.path.join(path, "coarse"))
        fine = {}
        for g, info in man["fine"].items():
            fine[g] = (info["single_label"] if info["single_label"] is not None
                       else ReferenceModel.load(os.path.join(path, f"fine_{g}")))
        return cls(coarse, fine, man["type_to_group"], man["classes"],
                   class_to_cl=man.get("class_to_cl"))


def build_hierarchical_reference(ref, label_key, embeddings=None, n_groups=8,
                                 hierarchy=None, ontology_key=None, **train_kwargs):
    """Build a HierarchicalReferenceModel from a labeled reference.

    Provide either ``embeddings`` (a per-cell array aligned to ``ref`` — the hierarchy is
    discovered from it) or a precomputed ``hierarchy`` dict ``{cell_type: group}`` (e.g.
    discovered from a QC-filtered embedding whose cells no longer align 1:1 to ``ref``).
    Any label in ``ref`` missing from ``hierarchy`` is placed in a catch-all group.
    Everything here is CPU; the embedding is the only step that may want a GPU.
    """
    ref = _load_adata(ref)
    labels = ref.obs[label_key].astype(str).to_numpy()
    if hierarchy is not None:
        grp = dict(hierarchy)
    elif embeddings is not None:
        grp = discover_hierarchy(embeddings, labels, n_groups)
    else:
        raise ValueError("provide either embeddings or a precomputed hierarchy")
    for t in set(labels) - set(grp):          # labels not covered by the hierarchy
        grp[t] = "_unmapped"
    ref.obs["_coarse"] = [grp[t] for t in labels]
    coarse = train_reference(ref, train_label_name="_coarse", **train_kwargs)
    fine = {}
    for g in sorted(set(grp.values())):
        sub = ref[ref.obs["_coarse"] == g]
        gtypes = sub.obs[label_key].unique()
        fine[g] = (str(gtypes[0]) if len(gtypes) == 1
                   else train_reference(sub.copy(), train_label_name=label_key, **train_kwargs))
    class_to_cl = None
    if ontology_key is not None and ontology_key in ref.obs:
        class_to_cl = {str(t): str(c) for t, c in
                       zip(labels, ref.obs[ontology_key].astype(str))}
    return HierarchicalReferenceModel(coarse, fine, grp, sorted(set(labels)),
                                      class_to_cl=class_to_cl)


def annotate(query, model, output_label_name="celltype", **kwargs):
    """Annotate a query AnnData with a (hierarchical or flat) reference model."""
    return model.predict(query, output_label_name=output_label_name, **kwargs)


def bundled_reference(name="broad_human_v1"):
    """Load a pre-trained HierarchicalReferenceModel shipped with actinn-jax.

    These annotate unknown human single-cell data out of the box on CPU — no scPRINT,
    no GPU. ``name`` resolves to ``actinn_jax/references/<name>/``.
    """
    path = os.path.join(os.path.dirname(__file__), "references", name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"bundled reference '{name}' not found at {path}")
    return HierarchicalReferenceModel.load(path)
