"""Time train + predict for the current JAX backend (CPU or Metal).

Run the same script under each venv to compare:
    .venv/bin/python        benchmark/bench_backend.py   # CPU
    .venv-metal/bin/python  benchmark/bench_backend.py   # Apple Metal GPU

Uses a fixed synthetic dataset so the two runs are directly comparable and need
no external files. Data construction happens before timing; only the model train
and predict calls are timed.
"""

import sys
import time

sys.path.insert(0, "tests")

import jax

import actinn_jax as ctp
from conftest import make_adata


def main():
    print("backend devices:", jax.devices())
    train = make_adata(n_per_type=2000, n_genes=2000, n_types=8, seed=1)
    query = make_adata(n_per_type=2000, n_genes=2000, n_types=8, seed=2)
    print(f"train: {train.shape}  query: {query.shape}")

    t = time.perf_counter()
    model = ctp.train_reference(train, train_label_name="celltype", print_cost=False)
    print(f"train_reference:        {time.perf_counter() - t:8.2f} s")

    for i in range(3):
        t = time.perf_counter()
        ctp.predict(query.copy(), model, output_label_name="pred")
        tag = "(cold/JIT)" if i == 0 else "(warm)"
        print(f"predict run {i + 1} {tag:11} {time.perf_counter() - t:8.3f} s")


if __name__ == "__main__":
    main()
