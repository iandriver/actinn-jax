"""Build your OWN hierarchical reference from labeled data (the offline, one-time step).

    python examples/build_reference.py labeled_reference.h5ad cell_type my_reference

`labeled_reference.h5ad` has raw counts + a cell-type column in `.obs`. scPRINT embeds
the reference once (GPU/Apple-MPS recommended; install `scprint` separately) to discover
a coarse->fine hierarchy; a small actinn-jax model is then trained on a HVG panel.
The saved model annotates new data on CPU with no scPRINT — see quickstart_annotate.py.
"""
import sys

import scanpy as sc

import actinn_jax as aj
from actinn_jax.embed import scprint_embed   # optional; needs `pip install scprint`

N_HVG = 4000
N_GROUPS = 8


def main(ref_path, label_key, out_dir, ckpt="medium-v1.5.ckpt"):
    ref = sc.read_h5ad(ref_path)
    print(f"reference {ref.shape} | {ref.obs[label_key].nunique()} cell types")

    # 1) embed the reference once (the only GPU step). Full gene set.
    emb = scprint_embed(ref, ckpt=ckpt)                 # (n_cells, 256)

    # 2) restrict the *trained* model to a HVG panel so it stays small/fast.
    raw = ref.copy()
    sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    sc.pp.highly_variable_genes(raw, n_top_genes=min(N_HVG, raw.n_vars))
    ref_hvg = ref[:, raw.var["highly_variable"].values].copy()

    # 3) discover hierarchy from embeddings + train coarse/fine models (CPU).
    model = aj.build_hierarchical_reference(ref_hvg, label_key, emb, n_groups=N_GROUPS,
                                            print_cost=False)
    model.save(out_dir)
    print(f"saved hierarchical reference to {out_dir}/  ({len(model.classes)} fine types, "
          f"{len(set(model.type_to_group.values()))} coarse groups)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cell_type",
         sys.argv[3] if len(sys.argv) > 3 else "my_reference")
