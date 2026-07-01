# actinn-jax

Fast, dependency-light cell-type **reference mapping** for single-cell data: train
on a labeled reference dataset, then annotate any query dataset.

## ⭐ Annotate any human single-cell data — fast, on CPU

A **two-stage workflow**: a foundation model (scPRINT) is used *once, offline* to
discover a coarse→fine cell-type hierarchy in a reference; a small actinn-jax model
trained on it then annotates new data in **milliseconds on a CPU — no GPU, no scPRINT
at inference**. A pre-trained broad-human reference sampled across the **whole CELLxGENE
census** (**~800 cell types, 314 tissues, 440 datasets** — the breadth scPRINT itself was
trained on) ships with the package, so unknown data works out of the box:

```python
import scanpy as sc, actinn_jax as aj

adata = sc.read_h5ad("my_human_data.h5ad")           # raw counts
model = aj.bundled_reference("broad_human_v1")        # pre-trained, CPU-only, ~800 types
adata = aj.annotate(adata, model, min_prob=0.5)       # -> obs['celltype', _coarse, _probability]
```

`min_prob` is an **abstain threshold**: cells whose confidence is below it are labeled
`"unknown"` instead of being force-mapped to the nearest reference type — so genuinely
novel cell types get flagged rather than mislabeled. Held-out calibration (10% of cell
types held out entirely as out-of-distribution):

| `min_prob` | accuracy (kept) | coverage | OOD flagged as unknown |
|-----------:|----------------:|---------:|-----------------------:|
| 0.0 (off)  | 0.55            | 100%     | 0%                     |
| 0.5        | 0.80            | 46%      | 75%                    |
| 0.7        | 0.86            | 30%      | 88%                    |

Breadth vs. precision is a deliberate trade: ~800-way annotation from a small per-type
sample is far harder than a narrow atlas (many near-duplicate subtypes, e.g. cortical
neuron layers), so raw accuracy is lower than a focused reference — the abstain threshold
recovers precision on the cells it keeps. For a narrower, higher-accuracy reference on
your own cell types, build one (below).

Or from the command line:

```bash
python examples/quickstart_annotate.py my_human_data.h5ad   # writes *_annotations.csv
```

**Build your own reference** (the only step that may use a GPU; install `scprint`):

```python
from actinn_jax.embed import scprint_embed
emb   = scprint_embed(ref_adata)                                   # GPU/MPS, once
model = aj.build_hierarchical_reference(ref_adata, "cell_type", emb)
model.save("my_reference")                                         # then annotate on CPU
```

See [`examples/`](examples) for both. The *why* — accuracy/speed/memory benchmarks and
the design rationale — lives in the companion
[actinn-jax-benchmark](https://github.com/iandriver/actinn-jax-benchmark) repo
([model-flow mini-paper](https://github.com/iandriver/actinn-jax-benchmark/blob/main/docs/MODEL_FLOW.md)).

---


`actinn-jax` is a from-scratch **JAX** reimplementation of
[ACTINN](https://github.com/mafeiyang/ACTINN.git) (Ma, Pellegrini et al.). The
original ACTINN was written in TensorFlow 1.x; this version replaces it with a
small JIT-compiled JAX MLP and a sparse-aware preprocessing pipeline, adds a
cached "train once, map many" reference model, and scales to atlas-sized data.

## Attribution

This package is based entirely on the ACTINN method. If you use it, please cite
the original work:

> Feiyang Ma, Matteo Pellegrini. **ACTINN: automated identification of cell types
> in single cell RNA sequencing.** *Bioinformatics*, 2020.
> https://doi.org/10.1093/bioinformatics/btz592

Original ACTINN implementation: https://github.com/mafeiyang/ACTINN

## Installation

```bash
git clone https://github.com/iandriver/actinn-jax.git
cd actinn-jax
pip install .
```

### Requirements

1. [JAX](https://docs.jax.dev/en/latest/installation.html) — CPU by default, runs everywhere
2. [scanpy](https://scanpy.readthedocs.io/) / [anndata](https://anndata.readthedocs.io/)

The classifier is a small MLP in JAX — **no TensorFlow dependency**. The model is
small enough that the **CPU path is fast and is recommended**. On Apple Silicon an
experimental GPU backend is available via `pip install .[metal]` (installs
`jax-metal`), but for this model size it is not generally faster than CPU.

## Input data

`actinn-jax` expects **raw integer counts**. For CELLxGENE-style objects where
`.X` is normalized and raw counts live in `.raw`, the default `use_raw='auto'`
picks the raw counts automatically. Genes are matched by `var_names`
(case-insensitive); the reference and query must share gene-name format.

## Usage

### One-off: train and predict in a single call

```python
import actinn_jax as aj

adata, params = aj.celltype_predict_actinn(
    adata,                       # query AnnData (raw counts)
    'reference.h5ad',            # labeled reference
    '/path/to/output_dir',
    train_label_name='cell_type',
    output_label_name='celltype_pred',
)
```

Predictions are written to `adata.obs['celltype_pred']` and confidences to
`adata.obs['celltype_pred_probability']`.

### Reference mapping: train once, map many queries (fastest)

```python
import actinn_jax as aj

# Train once and cache to disk (weights .npz + metadata .json).
model = aj.train_reference('reference.h5ad', train_label_name='cell_type')
model.save('/path/to/my_reference')

# Later / elsewhere: load and annotate any number of queries, no retraining.
model = aj.ReferenceModel.load('/path/to/my_reference')
adata, _ = aj.predict(adata, model, output_label_name='celltype_pred')
```

### Large / atlas-scale references

```python
# Balance + cap training cells per class (bounds memory, speeds up training);
# prediction runs in cell chunks so huge queries never fully densify.
model = aj.train_reference(
    'big_atlas.h5ad',
    train_label_name='cell_type',
    max_cells_per_label=500,     # subsample per class
)
adata, _ = aj.predict(adata, model, chunk_size=50000)
```

## How it works

- **Sparse-aware preprocessing** — gene-name matching and intersection happen on
  the sparse matrix; only the shared-gene submatrix is densified.
- **Normalization** follows ACTINN: per-cell library-size normalize to 1e4,
  `log2(x+1)`, then expr- and CV-percentile gene filtering.
- **Model** — a 4-layer MLP (100→50→25→n_types), Glorot init, Adam with
  exponential-decay schedule, trained with a JIT-compiled epoch step.
- **Cached `ReferenceModel`** stores the trained weights, gene set, and label
  mapping so queries are projected onto the reference's fixed gene space.

## Tests & benchmarks

```bash
pip install .[test]
pytest                              # synthetic-data unit tests
python benchmark/benchmark.py       # timing on synthetic data
```

## License

MIT. See [LICENSE](LICENSE).
