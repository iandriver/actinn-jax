"""Novel cell-type discovery — validated by holding out a real discovery.

We train a reference on the Krasnow et al. (2020) human lung atlas but WITHHOLD the
pulmonary ionocyte (itself a landmark 2018 novel-cell-type discovery), then ask
``detect_novel_celltypes`` to find what's missing. The withheld ionocytes should be
flagged as a novel population and nothing else should be.

    python examples/detect_novel_celltypes.py path/to/krasnow_lung_atlas_10x.h5ad

The atlas is available from the Human Lung Cell Atlas / CELLxGENE. Any h5ad with raw
counts and a cell-type column works — edit LABEL_COL / HOLDOUT below.
"""
import sys
import warnings; warnings.filterwarnings("ignore")

import scanpy as sc
import actinn_jax as aj

LABEL_COL = "free_annotation"      # cell-type annotation to build the reference from
HOLDOUT = "Ionocyte"               # withhold this type -> it becomes "novel"

path = sys.argv[1] if len(sys.argv) > 1 else "krasnow_lung_atlas_10x.h5ad"
adata = sc.read_h5ad(path)
adata.obs["label"] = adata.obs[LABEL_COL].astype(str)
n_holdout = int((adata.obs["label"] == HOLDOUT).sum())
print(f"{adata.n_obs:,} cells, {adata.obs.label.nunique()} types; "
      f"{HOLDOUT} = {n_holdout} cells")

# Reference that has never seen the held-out type.
train = adata[adata.obs["label"] != HOLDOUT].copy()
model = aj.train_reference(train, train_label_name="label",
                           max_cells_per_label=300, print_cost=False)

# Detect novel populations. Use the atlas's own annotation as the clustering so the
# result is easy to read; in practice pass your own Leiden/Louvain clusters, or omit
# cluster_key to compute one.
evidence, markers = aj.detect_novel_celltypes(
    model, adata, cluster_key=LABEL_COL, min_prob=0.5, min_cells=10)

flagged = evidence[evidence.novel]
print(f"\nflagged {len(flagged)} novel population(s):")
print(flagged[["cluster", "n_cells", "frac_low_conf", "median_conf",
               "nearest_label"]].to_string(index=False))

# Translate the top markers of the held-out type to gene symbols if available.
symcol = next((c for c in ("feature_name", "gene_name", "gene_symbol", "symbol")
               if c in adata.var.columns), None)
for cl, name in [(r["cluster"], f"novel_{i+1}")
                 for i, r in enumerate(flagged.to_dict("records"))]:
    genes = markers[name]
    if symcol:
        m = dict(zip(adata.var_names.astype(str), adata.var[symcol].astype(str)))
        genes = [m.get(g, g) for g in genes]
    print(f"  {name} (was '{cl}') markers: {genes}")
