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

**`refine_to_query`** masks the broad reference down to just the classes your query's own
predictions actually support (no ground truth, no retraining, sub-second):

```python
refined = aj.refine_to_query(model, adata)
adata = aj.annotate(adata, refined)
```

Measured on real data (see
[actinn-jax-benchmark's REFINE.md](https://github.com/iandriver/actinn-jax-benchmark/blob/main/docs/REFINE.md)):
this is a safe, free pruning pass — it reliably protects real types and never made
accuracy worse in testing — but it's **not a fix** for the large-reference accuracy gap.
The classes actually causing confusion are the model's genuinely-confusable siblings of
real types, and they carry the same confidence signature as real rare ones, so no
threshold built from the classifier's own output cleanly separates them. **Retraining on
a narrower, focused reference (`build_reference.py` above) is the reliable way to close
that gap** — it measurably outperforms masking because it reshapes the decision boundary
rather than restricting a frozen one's candidate set.

**`refine_to_tissue`** uses a stronger, label-free prior: a sample is (mostly) from one
tissue, and a liver sample should not be labeled with lung-specific epithelium. It prunes
the candidate classes to those the CELLxGENE census records in that tissue, while keeping
**pan-tissue** types (immune, endothelial, stromal) available everywhere — so
liver-resident T cells and macrophages are unaffected. The broad reference ships with a
per-class tissue map (`tissue_general` from ~62 M census cells) baked in, so this is
offline and instant:

```python
refined = aj.refine_to_tissue(model, tissue='liver')   # or ['liver','blood'] for mixed
adata = aj.annotate(adata, refined)
# or read the tissue from adata.obs['tissue'/'tissue_general'] automatically:
refined = aj.refine_to_tissue(model, adata=adata)      # tissue inferred from obs
```

On a real liver query this roughly **halves** the label set — it removes the cross-tissue
misfires (cardiac muscle, alveolar fibroblast, colonocyte, adrenal cortex…) that the broad
798-type model otherwise scatters in, while leaving hepatocyte / LSEC / stellate counts
essentially unchanged. It composes with `refine_to_query` (a class must be both evidenced
*and* tissue-plausible):

```python
refined = aj.refine_to_query(model, adata, tissue='liver')     # tissue='auto' reads obs
```

Common synonyms (`PBMC`→blood, `hepatic`→liver, …) are recognized; a tissue not in the
reference's vocabulary, or a reference with no tissue map, simply imposes no filter.

## Finding novel / unknown cell types

A cell type the reference has never seen doesn't show up as a few scattered uncertain
cells — it shows up as a **coherent group the reference can't confidently explain**.
`detect_novel_celltypes` looks for exactly that: clusters that are both large enough to
be a real population and predominantly low-confidence. (This is cluster-level rejection —
distinct from the per-cell `min_prob` abstain, which flags individual uncertain cells.)

```python
evidence, markers = aj.detect_novel_celltypes(
    model, adata,
    cluster_key='leiden',   # your clustering; omit to auto-cluster (needs leidenalg)
    min_cells=10,           # size floor for calling something novel — lower to go rarer
)
evidence[evidence.novel]    # candidate novel populations, most-suspect first
markers['novel_1']          # its top marker genes (vs the rest of the dataset)
adata.obs['novel']          # 'novel_1'… on flagged cells, '' otherwise
```

`min_cells` is the tunable **minimum population size** — the default (10) separates a real
novel type from noise/doublets; lower it (`min_cells=5`) to surface rarer candidates, raise
it to be stricter. `evidence` also reports each cluster's `nearest_label` (what the
reference *would* have called it) and `frac_low_conf`.

**Validated on a real discovery.** In the Krasnow *et al.* (2020) human lung atlas we
trained a reference but **withheld the pulmonary ionocyte** — itself a landmark 2018
novel-cell-type discovery (Montoro/Plasschaert) — then ran `detect_novel_celltypes`. Of all
57 populations, the 22 withheld ionocytes were the **only** cluster flagged (86% low
confidence, median 0.30; the reference would have mislabeled them "B cell"), and the
recovered markers were led by **ASCL3** (the ionocyte master transcription factor) and
**ATP6V0B** (the V-ATPase subunit ionocytes are named for). Raising `min_cells` above 22
correctly drops the flag — the knob does what it says. See
[`examples/detect_novel_celltypes.py`](examples/detect_novel_celltypes.py).

## Focused reference: liver (HLiCA)

`liver_hlica_v2` ships a **48-type, 7-lineage liver reference** built from
[HLiCA](https://doi.org/10.64898/2026.06.30.735539) (Edgar, Portman, Hu et al. 2026),
a 522,730-cell, 110-donor, expert-curated integrated human liver atlas — including
hepatocyte and endothelial **zonation** (periportal/pericentral), plasmacytoid dendritic
cells, and HLiCA's own novel-cell-type findings (NRXN1+ stromal cells, CUX2+ hepatic
stellate cells, MAMLD1+ trans monocytes, TREM2+ macrophages). This is exactly the
`refine_to_query` → `build_reference.py` recommendation above, executed for real: on a
held-out study, this focused reference reaches **exact-CL 0.72 / ontology 0.86**, vs.
**0.23 / 0.58** for the broad 798-type reference on the same cells (see
[actinn-jax-benchmark's HLICA_LIVER.md](https://github.com/iandriver/actinn-jax-benchmark/blob/main/docs/HLICA_LIVER.md)
and
[HLICA_EDGE_CASES.md](https://github.com/iandriver/actinn-jax-benchmark/blob/main/docs/HLICA_EDGE_CASES.md)
for the full cross-study validation and how v2 fixed two coverage gaps found by checking
the paper's own stated edge cases against v1).

```python
model = aj.bundled_reference("liver_hlica_v2")
adata = aj.annotate(adata, model)
```

(`liver_hlica_v1`, 38 types, also ships — for anyone who wants the smaller taxonomy.)

Data used under CC-BY 4.0; please cite the original paper if you use this reference.

**Runnable notebooks:**
[`examples/annotate_with_timing.ipynb`](examples/annotate_with_timing.ipynb) annotates a
65k-cell human lung atlas end-to-end — load, throughput (cells/s), exact vs
**ontology-aware** concordance, and the abstain sweep.
[`examples/liver_zonation.ipynb`](examples/liver_zonation.ipynb) shows **fine-label
mapping around hepatocyte zonation** — mapping a liver dataset onto the portal→mid→central
axis, with a confusion heatmap and ordinal (within-1-zone) scoring. See also
[`examples/quickstart_annotate.py`](examples/quickstart_annotate.py) and
[`examples/build_reference.py`](examples/build_reference.py). The *why* —
accuracy/speed/memory benchmarks and the design rationale — lives in the companion
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

The classifier is a small MLP in JAX — **no TensorFlow dependency** — so a plain
CPU install works everywhere and needs no special wheels.

### With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) resolves the scientific stack (JAX, scanpy) fast
and pins a compatible Python for you:

```bash
git clone https://github.com/iandriver/actinn-jax.git
cd actinn-jax
uv venv --python 3.12                 # create .venv on a supported Python
uv pip install -e .                   # install actinn-jax + deps into it
source .venv/bin/activate             # (or: `uv run python -c "import actinn_jax"`)
```

### With `pip`

```bash
git clone https://github.com/iandriver/actinn-jax.git
cd actinn-jax
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Published builds also install straight from PyPI once released: `pip install actinn-jax`.

### Running the example notebooks

The [`examples/`](examples/) notebooks need a few extra packages (Jupyter,
matplotlib, the `pronto` ontology reader). Install the `notebooks` extra and
launch JupyterLab:

```bash
uv pip install -e ".[notebooks]"      # or: pip install -e ".[notebooks]"
uv run jupyter lab examples/          # or: jupyter lab examples/
```

Each notebook has a `QUERY = '...'` line near the top — point it at your own
`.h5ad` (raw counts). No GPU or extra download is required: the pre-trained
references ship with the package.

### Requirements

1. [JAX](https://docs.jax.dev/en/latest/installation.html) — CPU by default, runs everywhere
2. [scanpy](https://scanpy.readthedocs.io/) / [anndata](https://anndata.readthedocs.io/)
3. Python 3.10–3.12 (3.13+ may lack JAX/scanpy wheels)

The model is small enough that the **CPU path is fast and is recommended**. On
Apple Silicon an experimental GPU backend is available via `uv pip install -e .[metal]`
(installs `jax-metal`), but for this model size it is not generally faster than CPU.

## Input data

`actinn-jax` expects **raw integer counts**. For CELLxGENE-style objects where
`.X` is normalized and raw counts live in `.raw`, the default `use_raw='auto'`
picks the raw counts automatically. Genes are matched by `var_names`
(case-insensitive); the reference and query must share gene-name format.

### Gene identifiers are matched automatically

Genes are matched by identifier, so query and reference must use the **same
identifier type**. The bundled references (`broad_human_v1`, `liver_hlica_v2`, …)
are keyed by **Ensembl gene IDs** (`ENSG…`). **You usually don't need to do
anything:** if your query's `var_names` are gene **symbols** (`A1BG`, `CD3D`, …)
and don't match, `actinn-jax` automatically falls back to the best-matching
`.var` column (`Ensembl_id`, `gene_ids`, `feature_id`, …) and prints a one-line
notice of which column it used.

If no identifier matches at all (neither `var_names` nor any `.var` column), it
raises a clear error rather than silently returning **one constant label** for
every cell (the classic symptom of an unmatched gene set). To force a specific
identifier yourself:

```python
adata.var_names = adata.var['Ensembl_id'].astype(str).values   # or 'gene_ids', etc.
adata.var_names_make_unique()
adata = aj.annotate(adata, model)
```

(Symbol-keyed references work fine too — just make the *query* resolvable to the
*reference*. Check overlap with `set(adata.var_names) & set(model.coarse.norm_genes)`.)

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

### Standardization (opt-in accuracy boost)

`standardize=True` z-scores each gene with the reference's frozen mean/std and
applies the same transform to every query — a cheap domain-alignment into the
reference feature space (the CPU-transferable idea behind scANVI+scArches).
Across the Open Problems `label_projection` datasets it lifts mean accuracy
+0.3 pt and macro-F1 +1.1 pt, with the largest gains on batch-shifted and
fine-grained references (e.g. Tabula Sapiens +1.8 pt accuracy, GTEx macro-F1
+7.9 pt). See `docs/PREPROC_ABLATION.md` in the benchmark repo.

```python
model = aj.train_reference('reference.h5ad', train_label_name='cell_type',
                           standardize=True)   # scaler saved with the model
```

It is **opt-in** (default `False`) because it shifts the softmax probability
calibration, which the two-stage refine / abstain thresholds are tuned against;
use it for a one-stage accuracy win, or re-tune those thresholds before combining.

## How it works

- **Sparse-aware preprocessing** — gene-name matching and intersection happen on
  the sparse matrix; only the shared-gene submatrix is densified.
- **Normalization** follows ACTINN: per-cell library-size normalize to 1e4,
  `log2(x+1)`, then expr- and CV-percentile gene filtering. Optional per-gene
  standardization (`standardize=True`) aligns the query into the reference's
  feature space for an accuracy boost.
- **Model** — a 4-layer MLP (100→50→25→n_types), Glorot init, Adam with
  exponential-decay schedule, trained with a JIT-compiled epoch step.
- **Cached `ReferenceModel`** stores the trained weights, gene set, and label
  mapping so queries are projected onto the reference's fixed gene space.

## Tests & benchmarks

```bash
uv pip install -e ".[test]"         # or: pip install -e ".[test]"
pytest                              # synthetic-data unit tests
python benchmark/benchmark.py       # timing on synthetic data
```

## License

MIT. See [LICENSE](LICENSE).
