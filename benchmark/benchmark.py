"""Benchmark the JAX ACTINN pipeline.

Times the cost of a one-off train+predict versus the amortized cached-reference
path (train once, then map many queries with no retraining).

Usage:
    python benchmark/benchmark.py                      # synthetic data
    python benchmark/benchmark.py REF.h5ad QUERY.h5ad LABEL_COL
"""

import sys
import time

sys.path.insert(0, "tests")  # reuse the synthetic data generator

import scanpy as sc

import actinn_jax as ctp


class timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t
        print(f"  {self.label:<34} {self.dt:8.3f} s")


def get_data():
    if len(sys.argv) >= 3:
        ref = sc.read_h5ad(sys.argv[1])
        query = sc.read_h5ad(sys.argv[2])
        label = sys.argv[3] if len(sys.argv) > 3 else "celltype"
        return ref, query, label
    from conftest import make_adata
    print("(no data passed -- using synthetic 4-type dataset)")
    return make_adata(n_per_type=400, seed=1), make_adata(n_per_type=400, seed=2), "celltype"


def main():
    ref, query, label = get_data()
    print(f"reference: {ref.shape}   query: {query.shape}   label: {label!r}\n")

    print("Cached-reference path (train once, reuse):")
    with timer("train_reference (one time)"):
        model = ctp.train_reference(ref, train_label_name=label, print_cost=False)
    with timer("save"):
        import tempfile, os
        model.save(os.path.join(tempfile.mkdtemp(), "ref"))
    # Warm + repeated query mapping shows the amortized per-query cost.
    for i in range(3):
        with timer(f"predict query (run {i + 1})"):
            ctp.predict(query.copy(), model, output_label_name="pred")

    print()


if __name__ == "__main__":
    main()
