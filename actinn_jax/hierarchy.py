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
import warnings

import numpy as np
import pandas as pd

from .actinn_predict import (
    ReferenceModel, train_reference, _load_adata, _select_gene_ids,
)


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


def _masked_argmax(proba, classes, allowed=None):
    """argmax over ``proba`` (n, C), optionally restricted+renormalized to ``allowed``.

    Zeroing the disallowed columns of an already-softmaxed row and renormalizing over
    the survivors is algebraically identical to computing softmax restricted to that
    column subset in the first place (the shared normalizer cancels) — so this needs no
    retraining, just the model's own output probabilities.
    """
    classes = np.asarray(classes)
    if allowed is None:
        idx = np.argmax(proba, axis=1)
        return classes[idx], proba[np.arange(len(idx)), idx]
    mask = np.array([c in allowed for c in classes])
    if not mask.any():
        return np.full(proba.shape[0], "unknown", dtype=object), np.zeros(proba.shape[0], dtype=np.float32)
    sub = proba[:, mask]
    sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
    idx = np.argmax(sub, axis=1)
    return classes[mask][idx], sub[np.arange(len(idx)), idx]


class HierarchicalReferenceModel:
    """A coarse ReferenceModel + one fine ReferenceModel per coarse group."""

    def __init__(self, coarse, fine, type_to_group, classes, class_to_cl=None,
                 class_to_tissue=None):
        self.coarse = coarse                  # ReferenceModel over coarse groups
        self.fine = fine                      # {group: ReferenceModel | single-label str}
        self.type_to_group = dict(type_to_group)
        self.classes = list(classes)
        # optional {cell_type_name: Cell-Ontology id} for ontology-aware roll-up/eval
        self.class_to_cl = dict(class_to_cl) if class_to_cl else {}
        # optional {cell_type_name: [tissue_general, ...]} for tissue-aware refine.
        # ["*"] means pan-tissue (allowed everywhere); a class absent from the map
        # is likewise treated as always-allowed. See refine_to_tissue().
        self.class_to_tissue = dict(class_to_tissue) if class_to_tissue else {}

    def _resolve_gene_ids(self, adata, use_raw):
        """Resolve the query's gene identifier against the reference once, warning
        at most once, so the many coarse/fine sub-model calls don't each re-match
        (and re-warn). Returns the ``gene_names`` override (or ``None``)."""
        col, idx = _select_gene_ids(adata, self.coarse.norm_genes, use_raw)
        if col is not None:
            warnings.warn(
                f"actinn-jax: query var_names matched few reference genes; using "
                f"adata.var['{col}'] instead (better overlap with the reference).",
                stacklevel=3,
            )
        return idx

    def predict_frame(self, adata, use_raw="auto", chunk_size=50000, min_prob=None,
                       allowed_groups=None, allowed_classes=None):
        """Coarse-predict, route each cell to its group's fine model, return a frame.

        ``min_prob`` (0-1, optional): cells whose final fine-label probability is below
        this are relabeled ``"unknown"`` (abstain), so out-of-distribution cells — types
        not in the reference — are flagged rather than force-labeled.

        ``allowed_groups`` / ``allowed_classes`` (optional, from :func:`refine_to_query`):
        restrict the coarse / per-group-fine softmax to these classes before taking the
        argmax (renormalized) — sharpens calls when most of the ~hundreds of shipped
        classes are known to be absent from this dataset. ``allowed_classes`` is
        ``{group_id: set(class names)}``.
        """
        gene_names = self._resolve_gene_ids(adata, use_raw)
        cproba = self.coarse.predict_proba(adata, use_raw=use_raw, chunk_size=chunk_size,
                                           gene_names=gene_names, _skip_match=True)
        coarse_pred, coarse_prob = _masked_argmax(cproba, self.coarse.classes, allowed_groups)
        out = np.empty(adata.n_obs, dtype=object)
        prob = np.zeros(adata.n_obs, dtype=np.float32)
        for g in np.unique(coarse_pred):
            mask = coarse_pred == g
            if g == "unknown":
                out[mask] = "unknown"
                continue
            fm = self.fine.get(str(g))
            if fm is None:
                out[mask] = "unknown"
            elif isinstance(fm, str):           # group has a single fine type
                out[mask] = fm
                prob[mask] = coarse_prob[mask]
            else:
                allowed = allowed_classes.get(str(g)) if allowed_classes else None
                fproba = fm.predict_proba(adata[mask], use_raw=use_raw, chunk_size=chunk_size,
                                          gene_names=gene_names, _skip_match=True)
                flab, fprob = _masked_argmax(fproba, fm.classes, allowed)
                out[mask] = flab
                prob[mask] = fprob
        if min_prob is not None:
            out[prob < min_prob] = "unknown"
        return pd.DataFrame({"celltype": out, "celltype_probability": prob,
                             "coarse": coarse_pred}, index=list(adata.obs_names)), None

    def predict(self, adata, output_label_name="celltype", use_raw="auto",
                chunk_size=50000, min_prob=None, allowed_groups=None, allowed_classes=None):
        """Annotate ``adata`` in place: adds ``celltype``, ``_probability``, ``_coarse``."""
        frame, _ = self.predict_frame(adata, use_raw=use_raw, chunk_size=chunk_size,
                                      min_prob=min_prob, allowed_groups=allowed_groups,
                                      allowed_classes=allowed_classes)
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
                       "class_to_cl": self.class_to_cl,
                       "class_to_tissue": self.class_to_tissue, "fine": fine_manifest}, fh)
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
                   class_to_cl=man.get("class_to_cl"),
                   class_to_tissue=man.get("class_to_tissue"))


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


def detect_present_classes(model, adata, use_raw="auto", chunk_size=50000,
                           min_mass=1.0, min_top1=1, top1_conf=0.3):
    """Estimate which of a broad model's classes are actually evidenced in ``adata``.

    Uses only the model's own (unmasked) predictions on this dataset — no ground truth,
    no retraining. A class is kept if *either*:

    - ``mass`` (its probability summed over every query cell) >= ``min_mass`` — catches
      real types that never individually win but are consistently a plausible runner-up
      across many cells (broad, diffuse evidence), or
    - it wins outright (argmax) for >= ``min_top1`` cells with probability >= ``top1_conf``
      — catches real but rare types that are unambiguous for a handful of cells even
      though they never accumulate much aggregate mass.

    This dual criterion is deliberately permissive: a class with *no* evidence of either
    kind anywhere in the dataset is dropped; anything with real (even weak-but-broad, or
    rare-but-clear) support survives. Coarse groups with zero surviving fine classes are
    dropped too. A class that is truly absent, and never even weakly competitive for any
    cell, cannot be recovered by this or any purely data-driven method.

    Returns
    -------
    allowed_groups : set of str
    allowed_classes : {group_id: set(class name)}
    evidence : pandas.DataFrame (one row per group/class) with mass, top1_count,
        max_prob, kept — for inspection/tuning.
    """
    gene_names = model._resolve_gene_ids(adata, use_raw)
    cproba = model.coarse.predict_proba(adata, use_raw=use_raw, chunk_size=chunk_size,
                                        gene_names=gene_names, _skip_match=True)
    coarse_classes = np.asarray(model.coarse.classes)
    coarse_pred = coarse_classes[np.argmax(cproba, axis=1)]

    rows = []
    allowed_classes = {}
    for g in np.unique(coarse_pred):
        fm = model.fine.get(str(g))
        mask = coarse_pred == g
        if fm is None:
            continue
        if isinstance(fm, str):
            allowed_classes[str(g)] = {fm}
            rows.append({"group": g, "class": fm, "mass": float(mask.sum()),
                         "top1_count": int(mask.sum()), "max_prob": 1.0, "kept": True})
            continue
        fproba = fm.predict_proba(adata[mask], use_raw=use_raw, chunk_size=chunk_size,
                                  gene_names=gene_names, _skip_match=True)
        classes = np.asarray(fm.classes)
        mass = fproba.sum(axis=0)
        top1 = np.argmax(fproba, axis=1)
        top1_count = np.bincount(top1, minlength=len(classes))
        max_prob = np.zeros(len(classes))
        for ci in np.unique(top1):
            max_prob[ci] = fproba[top1 == ci, ci].max()
        keep = (mass >= min_mass) | ((top1_count >= min_top1) & (max_prob >= top1_conf))
        kept_here = set(classes[keep])
        if kept_here:
            allowed_classes[str(g)] = kept_here
        rows.extend({"group": g, "class": c, "mass": float(m), "top1_count": int(t1),
                    "max_prob": float(mp), "kept": bool(k)}
                    for c, m, t1, mp, k in zip(classes, mass, top1_count, max_prob, keep))
    return set(allowed_classes), allowed_classes, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tissue-aware refinement
# --------------------------------------------------------------------------- #
# Map common user tissue names onto the census ``tissue_general`` vocabulary.
_TISSUE_SYNONYMS = {
    "pbmc": "blood", "peripheral blood": "blood", "whole blood": "blood",
    "peripheral blood mononuclear cell": "blood",
    "gut": "intestine", "bowel": "intestine", "cns": "brain",
    "renal": "kidney", "hepatic": "liver", "pulmonary": "lung", "cardiac": "heart",
}


def _known_tissues(model):
    """The set of tissue_general categories present in the model's tissue map."""
    out = set()
    for v in model.class_to_tissue.values():
        if v != ["*"]:
            out.update(v)
    return out


def _normalize_tissue(t, known):
    """Map a free-text tissue name onto a known tissue_general category, or None."""
    s = str(t).strip().lower()
    if s in known:
        return s
    if _TISSUE_SYNONYMS.get(s) in known:
        return _TISSUE_SYNONYMS[s]
    for k in known:                       # a known category appearing as a word
        if k == s or k in s.split():
            return k
    return None


def _infer_tissue(adata):
    """Read tissue label(s) from ``adata.obs`` (``tissue_general`` or ``tissue``)."""
    for col in ("tissue_general", "tissue", "Tissue", "organ"):
        if col in adata.obs:
            vals = adata.obs[col].astype(str)
            return [v for v in vals.unique() if v and v.lower() != "nan"]
    return []


def _resolve_tissues(model, tissue, adata):
    """Return the set of normalized tissue_general categories to filter to, or None.

    ``tissue`` may be a name, a list of names, ``'auto'`` (infer from
    ``adata.obs``), or ``None`` (no tissue filter).
    """
    if not model.class_to_tissue:
        warnings.warn("actinn-jax: this reference has no tissue map; tissue "
                      "filtering is a no-op.", stacklevel=3)
        return None
    if tissue is None:
        return None
    if isinstance(tissue, str) and tissue.lower() == "auto":
        raw = _infer_tissue(adata) if adata is not None else []
        if not raw:
            warnings.warn("actinn-jax: tissue='auto' but no tissue column found in "
                          "adata.obs; no tissue filter applied.", stacklevel=3)
            return None
        warnings.warn(f"actinn-jax: inferred tissue {sorted(set(raw))} from adata.obs.",
                      stacklevel=3)
    else:
        raw = [tissue] if isinstance(tissue, str) else list(tissue)
    known = _known_tissues(model)
    norm = {n for n in (_normalize_tissue(t, known) for t in raw) if n}
    unmatched = [t for t in raw if _normalize_tissue(t, known) is None]
    if unmatched:
        warnings.warn(f"actinn-jax: tissue(s) {unmatched} not in the reference's "
                      "tissue vocabulary; they impose no filter.", stacklevel=3)
    return norm or None


def _tissue_allowed_names(model, tissues):
    """Class names allowed for ``tissues``: pan-tissue (``['*']``) and unmapped
    classes always pass; organ-specific classes pass only if one of ``tissues`` is
    in their recorded set."""
    allow = set()
    for c in model.classes:
        t = model.class_to_tissue.get(c)
        if t is None or t == ["*"] or any(x in t for x in tissues):
            allow.add(c)
    return allow


def _names_to_allowed(model, names):
    """Convert an allowed class-name set into ``(allowed_groups, allowed_classes)``."""
    allowed_classes = {}
    for c in names:
        allowed_classes.setdefault(str(model.type_to_group.get(c)), set()).add(c)
    return set(allowed_classes), allowed_classes


def refine_to_tissue(model, tissue=None, adata=None):
    """Restrict a broad reference to the cell types plausible in a given tissue.

    A sample is (mostly) from one tissue, and a liver sample should not be labeled
    with lung-specific epithelium. This prunes the candidate classes to those the
    census records in ``tissue`` — while keeping pan-tissue types (immune,
    endothelial, stromal) available everywhere, so liver-resident T cells and
    macrophages are unaffected. No ground truth, no retraining.

        refined = aj.refine_to_tissue(model, tissue='liver')
        adata = refined.predict(adata)

    ``tissue`` is a name (or list) from the census ``tissue_general`` vocabulary
    ('liver', 'lung', 'blood', 'heart', …); common synonyms like 'PBMC'→blood are
    recognized. If ``tissue`` is None and ``adata`` is given, the tissue is read
    from ``adata.obs['tissue'/'tissue_general']``.
    """
    if tissue is None and adata is not None:
        tissue = "auto"
    tissues = _resolve_tissues(model, tissue, adata)
    if not tissues:
        return RefinedReference(model, None, None, None)      # no-op filter
    groups, classes = _names_to_allowed(model, _tissue_allowed_names(model, tissues))
    return RefinedReference(model, groups, classes, None)


class RefinedReference:
    """A view over a :class:`HierarchicalReferenceModel` restricted to classes actually
    evidenced in one query dataset. Built by :func:`refine_to_query`; wraps the same
    trained weights (no retraining, no reference data needed) — see its docstring."""

    def __init__(self, model, allowed_groups, allowed_classes, evidence):
        self.model = model
        self.allowed_groups = allowed_groups
        self.allowed_classes = allowed_classes
        self.evidence = evidence

    def predict_frame(self, adata, **kwargs):
        return self.model.predict_frame(adata, allowed_groups=self.allowed_groups,
                                        allowed_classes=self.allowed_classes, **kwargs)

    def predict(self, adata, output_label_name="celltype", **kwargs):
        return self.model.predict(adata, output_label_name=output_label_name,
                                  allowed_groups=self.allowed_groups,
                                  allowed_classes=self.allowed_classes, **kwargs)


def refine_to_query(model, adata, tissue=None, use_raw="auto", chunk_size=50000,
                    min_mass=1.0, min_top1=1, top1_conf=0.3):
    """Mask a broad reference down to the classes evidenced in ``adata`` — no retraining.

    Measured on real ground-truth queries (see actinn-jax-benchmark's ``docs/REFINE.md``):
    this is a **safe, free pruning pass, not an accuracy fix**. It reliably protects real
    types (rarely drops one that's genuinely present) and never made accuracy *worse* in
    testing, but it also rarely makes accuracy meaningfully *better* — the handful of
    classes actually causing confusion tend to be the model's genuinely-confusable
    siblings of real types, which carry the same confidence signature as real rare types
    and so survive any threshold built from the classifier's own output. Closing that gap
    needs retraining on a narrower label set (see ``examples/build_reference.py``), which
    measurably outperforms masking because it reshapes the decision boundary rather than
    just restricting the candidate set of a frozen one.

        refined = aj.refine_to_query(model, adata)
        adata = refined.predict(adata)                 # same-or-better, never worse in testing
        refined.evidence.sort_values('mass', ascending=False).head()  # inspect the evidence

    Tune ``min_mass`` / ``min_top1`` / ``top1_conf`` to trade recall of rare true types
    against how aggressively implausible classes are pruned (defaults favor recall).

    ``tissue`` (optional) additionally restricts to cell types plausible in that
    tissue (see :func:`refine_to_tissue`): a name/list ('liver', 'lung', …), or
    ``'auto'`` to read it from ``adata.obs['tissue'/'tissue_general']``. The two
    filters compose (a class must be both evidenced *and* tissue-plausible). Left
    ``None`` (default), no tissue filter is applied.
    """
    allowed_groups, allowed_classes, evidence = detect_present_classes(
        model, adata, use_raw=use_raw, chunk_size=chunk_size,
        min_mass=min_mass, min_top1=min_top1, top1_conf=top1_conf)
    tissues = _resolve_tissues(model, tissue, adata)
    if tissues:
        keep = _tissue_allowed_names(model, tissues)
        allowed_classes = {g: (cs & keep) for g, cs in allowed_classes.items()}
        allowed_classes = {g: cs for g, cs in allowed_classes.items() if cs}
        allowed_groups = set(allowed_classes)
    return RefinedReference(model, allowed_groups, allowed_classes, evidence)


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
