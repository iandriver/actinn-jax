"""Verify ACTINN reference mapping on real lung atlases.

Experiments
-----------
split  : stratified held-out test within one dataset -> clean accuracy vs truth.
cross  : train on a (subsampled) reference atlas, predict a query atlas; report
         label concordance and per-stage timing. Reference is loaded in backed
         mode and subsampled before materializing, so atlas-scale .h5ad files
         never fully enter memory.

Examples
--------
  python benchmark/verify_real.py split  ~/Downloads/krasnow_lung_atlas_10x.h5ad
  python benchmark/verify_real.py cross \
      ~/Downloads/Sikkama_HCLA.h5ad ~/Downloads/krasnow_lung_atlas_10x.h5ad
"""

import argparse
import time

import numpy as np
import scanpy as sc

import actinn_jax as ctp

LABEL = "cell_type"
CL_ID = "cell_type_ontology_term_id"
DEFAULT_OBO = "/tmp/cl-basic.obo"


def load_cl_ancestors(obo_path):
    """Return ``{CL_id: frozenset(ancestor ids incl. self)}`` from a Cell Ontology OBO."""
    import pronto
    ont = pronto.Ontology(obo_path)
    anc = {}
    for term in ont.terms():
        anc[term.id] = frozenset(t.id for t in term.superclasses(with_self=True))
    return anc


def ontology_concordance(truth_cl, pred_cl, anc):
    """Fraction of cells whose predicted CL term is lineage-related to the truth.

    Lineage-related = identical, or one is an ancestor of the other (credits
    coarser/finer annotations and synonyms that share a path in the ontology).
    """
    ok = 0
    n = 0
    for t, p in zip(truth_cl, pred_cl):
        if not isinstance(t, str) or not isinstance(p, str) or not t or not p:
            continue
        n += 1
        if t == p or p in anc.get(t, ()) or t in anc.get(p, ()):
            ok += 1
    return ok / max(n, 1), n


class timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"  [start] {self.label} ...", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"  [done ] {self.label}: {time.perf_counter() - self.t:.2f} s", flush=True)


def stratified_subsample(labels, n_per_label, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > n_per_label:
            idx = rng.choice(idx, n_per_label, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def load_subsampled(path, label, n_per_label):
    """Backed read + stratified subsample, materializing only the subset."""
    backed = sc.read_h5ad(path, backed="r")
    labels = np.asarray(backed.obs[label].values)
    sel = stratified_subsample(labels, n_per_label)
    sub = backed[sel].to_memory()
    backed.file.close()
    return sub


def report_concordance(truth, pred, max_rows=15):
    truth, pred = np.asarray(truth, dtype=object), np.asarray(pred, dtype=object)
    acc = float((truth == pred).mean())
    print(f"\n  exact-label concordance: {acc:.3f}  ({len(truth)} cells)")
    # Most common (truth -> pred) mappings, to gauge sensible biology.
    import pandas as pd
    tab = pd.crosstab(pd.Series(truth, name="truth"), pd.Series(pred, name="pred"))
    top = tab.stack().sort_values(ascending=False).head(max_rows)
    print("  top (truth -> predicted) cell counts:")
    for (t, p), n in top.items():
        flag = "==" if t == p else "  "
        print(f"    {flag} {str(t)[:34]:<34} -> {str(p)[:34]:<34} {n}")
    return acc


def run_split(path, n_per_label, test_frac, seed):
    print(f"\n=== SPLIT experiment on {path.split('/')[-1]} ===")
    with timer("load + subsample"):
        adata = load_subsampled(path, LABEL, n_per_label)
    labels = np.asarray(adata.obs[LABEL].values)

    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        test_mask[rng.choice(idx, max(1, int(len(idx) * test_frac)), replace=False)] = True
    train_ad, test_ad = adata[~test_mask].copy(), adata[test_mask].copy()
    print(f"  train cells: {train_ad.n_obs}   test cells: {test_ad.n_obs}")

    with timer("train_reference"):
        model = ctp.train_reference(train_ad, train_label_name=LABEL, print_cost=False)
    with timer("predict (held-out)"):
        out, _ = ctp.predict(test_ad, model, output_label_name="pred")
    report_concordance(out.obs[LABEL].values, out.obs["pred"].values)


def run_cross(ref_path, query_path, n_per_label, query_cap, seed, obo_path=None):
    print(f"\n=== CROSS experiment: ref={ref_path.split('/')[-1]} "
          f"query={query_path.split('/')[-1]} ===")
    with timer(f"load reference (<= {n_per_label}/label, backed)"):
        ref = load_subsampled(ref_path, LABEL, n_per_label)
    print(f"  reference cells: {ref.n_obs}   types: {ref.obs[LABEL].nunique()}")
    # Reference label -> CL ontology id, for ontology-aware scoring later.
    name_to_cl = {}
    if CL_ID in ref.obs:
        name_to_cl = dict(zip(ref.obs[LABEL].astype(str), ref.obs[CL_ID].astype(str)))
    with timer("train_reference"):
        model = ctp.train_reference(ref, train_label_name=LABEL, print_cost=False)
    del ref

    with timer("load query"):
        query = (load_subsampled(query_path, LABEL, query_cap)
                 if query_cap else sc.read_h5ad(query_path))
    print(f"  query cells: {query.n_obs}")
    with timer("predict query (chunked)"):
        out, _ = ctp.predict(query, model, output_label_name="pred")
    report_concordance(out.obs[LABEL].values, out.obs["pred"].values)

    # Ontology-aware (lineage) concordance: credits coarser/finer + synonyms.
    if obo_path and name_to_cl and CL_ID in out.obs:
        with timer("load Cell Ontology"):
            anc = load_cl_ancestors(obo_path)
        truth_cl = out.obs[CL_ID].astype(str).values
        pred_cl = np.array([name_to_cl.get(p, "") for p in out.obs["pred"].astype(str)])
        acc, n = ontology_concordance(truth_cl, pred_cl, anc)
        exact = float((truth_cl == pred_cl).mean())
        print(f"\n  ONTOLOGY-AWARE concordance: {acc:.3f}  ({n} cells with CL ids)")
        print(f"  (exact CL-id match for comparison: {exact:.3f})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("split")
    sp.add_argument("path")
    sp.add_argument("--n-per-label", type=int, default=2000)
    sp.add_argument("--test-frac", type=float, default=0.25)
    sp.add_argument("--seed", type=int, default=0)

    cp = sub.add_parser("cross")
    cp.add_argument("ref_path")
    cp.add_argument("query_path")
    cp.add_argument("--n-per-label", type=int, default=500)
    cp.add_argument("--query-cap", type=int, default=0, help="0 = use full query")
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--obo", default=DEFAULT_OBO, help="Cell Ontology OBO for lineage scoring")

    a = p.parse_args()
    if a.mode == "split":
        run_split(a.path, a.n_per_label, a.test_frac, a.seed)
    else:
        run_cross(a.ref_path, a.query_path, a.n_per_label, a.query_cap or 0,
                  a.seed, a.obo)


if __name__ == "__main__":
    main()
