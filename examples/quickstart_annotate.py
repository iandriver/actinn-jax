"""Quickstart: annotate ANY unknown human single-cell dataset, fast, on CPU.

    python examples/quickstart_annotate.py my_data.h5ad

`my_data.h5ad` is a human scRNA-seq AnnData with **raw counts** (in `.X`, or in `.raw`
for CELLxGENE-style objects). Gene names should be Ensembl IDs or symbols. No GPU and
no scPRINT needed — annotation uses the bundled hierarchical reference on CPU.
"""
import sys

import scanpy as sc

import actinn_jax as aj


def main(path):
    adata = sc.read_h5ad(path)
    print(f"loaded {adata.shape}")

    model = aj.bundled_reference("broad_human_v1")     # pre-trained, CPU-only, ~800 types
    # min_prob is an abstain threshold: cells below it are labeled "unknown" instead of
    # being force-mapped to a reference type (recommended for out-of-distribution data).
    # 0.5 keeps high-confidence calls; lower it for more coverage, raise it for precision.
    adata = aj.annotate(adata, model, min_prob=0.5)    # adds obs['celltype', _coarse, _probability]

    cols = ["celltype", "celltype_coarse", "celltype_probability"]
    print(adata.obs[cols].head(10).to_string())
    n_unknown = (adata.obs["celltype"] == "unknown").sum()
    print(f"\n{n_unknown}/{adata.n_obs} cells below threshold -> 'unknown'")
    print("top predicted cell types:")
    print(adata.obs["celltype"].value_counts().head(15).to_string())

    out = path.rsplit(".h5ad", 1)[0] + "_annotations.csv"
    adata.obs[cols].to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "my_data.h5ad")
