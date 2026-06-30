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

    model = aj.bundled_reference("broad_human_v1")     # pre-trained, CPU-only
    adata = aj.annotate(adata, model)                  # adds obs['celltype', _coarse, _probability]

    cols = ["celltype", "celltype_coarse", "celltype_probability"]
    print(adata.obs[cols].head(10).to_string())
    print("\ntop predicted cell types:")
    print(adata.obs["celltype"].value_counts().head(15).to_string())

    out = path.rsplit(".h5ad", 1)[0] + "_annotations.csv"
    adata.obs[cols].to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "my_data.h5ad")
