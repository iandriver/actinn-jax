# actinn-jax

Fast, dependency-light cell-type **reference mapping** for single-cell data: train
on a labeled reference dataset, then annotate any query dataset.

## ⭐ Annotate single-cell data — fast, on CPU

Pre-trained organism-wide references ship with the package, so unknown data works out of
the box. From the command line, with no arguments to choose:

```bash
actinn-jax annotate my_data.h5ad --min-prob 0.5 -o annotations.csv
```

That infers the organism from the query's gene ids and loads the matching reference.
`actinn-jax references` lists what is installed. From Python:

```python
import scanpy as sc, actinn_jax as aj

for r in aj.list_references():                        # what's available, and for what
    print(r["name"], r["organism"], r["n_classes"])

adata = sc.read_h5ad("my_data.h5ad")                  # raw counts
model = aj.bundled_reference("panhuman_distill_v1")   # pre-trained, CPU-only
adata = aj.annotate(adata, model, min_prob=0.5)       # -> obs['celltype', _coarse, _probability]
```

Annotating with a reference for the wrong organism raises rather than returning
meaningless labels.

### Where the references live

`panhuman_distill_v1` (16 MB, human) ships inside the wheel, so `pip install actinn-jax`
annotates human data offline with nothing to download. The census-scale references are
tens of MB each and are **fetched once on first use** from Zenodo
([10.5281/zenodo.21688151](https://doi.org/10.5281/zenodo.21688151)), then cached in
`~/.cache/actinn-jax/` — `bundled_reference("broad_mouse_v1")` works whether or not it has
been downloaded before (~40 s on a cold cache, instant after). Each archive is
checksum-verified before it is unpacked.

For a tutorial or a workshop on a slow connection, pull them ahead of time:

```bash
actinn-jax fetch                      # everything
actinn-jax fetch broad_mouse_v1       # or just one
actinn-jax references                 # installed / cached / available
actinn-jax cache --clear              # free the space again
```

In a notebook the same thing is one call, and it is safe to re-run:

```python
from actinn_jax import registry
registry.fetch_reference("broad_mouse_v1")     # no-op once cached
```

Point `ACTINN_JAX_REFERENCE_URL` at a mirror to serve them from somewhere else, and
`ACTINN_JAX_CACHE` to relocate the cache (useful on a shared cluster filesystem).

Under the hood this is a **two-stage workflow**: a coarse→fine cell-type hierarchy is
discovered *once, offline*, and a small actinn-jax model trained on it annotates new data
in **milliseconds on a CPU — no GPU at inference**.

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

The abstain threshold above is for the ~800-type census reference; each bundled reference
reports its own calibration in `build_info.json`, and `broad_mouse_v1`'s is notably better
(0.88 accuracy at 82% coverage).

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
correctly drops the flag — the knob does what it says. The full walkthrough (with the
t-SNE highlight of the flagged population) is in
[`examples/novel_celltypes.ipynb`](https://github.com/iandriver/actinn-jax/blob/main/examples/novel_celltypes.ipynb); a script version is
[`examples/detect_novel_celltypes.py`](https://github.com/iandriver/actinn-jax/blob/main/examples/detect_novel_celltypes.py).

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
[`examples/annotate_with_timing.ipynb`](https://github.com/iandriver/actinn-jax/blob/main/examples/annotate_with_timing.ipynb) annotates a
65k-cell human lung atlas end-to-end — load, throughput (cells/s), exact vs
**ontology-aware** concordance, and the abstain sweep.
[`examples/liver_zonation.ipynb`](https://github.com/iandriver/actinn-jax/blob/main/examples/liver_zonation.ipynb) shows **fine-label
mapping around hepatocyte zonation** — mapping a liver dataset onto the portal→mid→central
axis, with a confusion heatmap and ordinal (within-1-zone) scoring. See also
[`examples/quickstart_annotate.py`](https://github.com/iandriver/actinn-jax/blob/main/examples/quickstart_annotate.py) and
[`examples/build_reference.py`](https://github.com/iandriver/actinn-jax/blob/main/examples/build_reference.py). The *why* —
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

The [`examples/`](https://github.com/iandriver/actinn-jax/blob/main/examples/) notebooks need a few extra packages (Jupyter,
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

## API reference

Every public name, with its parameters. The examples assume `import actinn_jax as aj`.

- [Training and prediction](#training-and-prediction) — `train_reference`, `predict`, `celltype_predict_actinn`, `ReferenceModel`
- [Two-stage references](#two-stage-references) — `build_hierarchical_reference`, `discover_hierarchy`, `annotate`, `HierarchicalReferenceModel`
- [Pre-trained references](#pre-trained-references) — `bundled_reference`, `list_references`, `default_reference_for`, `infer_organism`
- [Query-adaptive refinement](#query-adaptive-refinement) — `refine_to_query`, `refine_to_tissue`, `detect_present_classes`, `RefinedReference`
- [Novel cell types](#novel-cell-types) — `detect_novel_celltypes`
- [Submodules](#submodules) — `actinn_predict`, `hierarchy`, `actinn_utils`

### Training and prediction

#### `train_reference`

```python
train_reference(train_data, train_label_name="celltype", learning_rate=0.0001,
                num_epochs="auto", batch_size="auto", use_raw="auto",
                max_cells_per_label=None, seed=3, print_cost=True,
                standardize=False)
```

Train a classifier on a labelled reference. Returns a [`ReferenceModel`](#referencemodel).

- `train_data` — `.h5ad` path or `AnnData` holding raw counts (in `.raw` or `.X`) and cell-type labels in `obs[train_label_name]`.
- `train_label_name` — column of `obs` holding the labels.
- `learning_rate` — Adam learning rate, used with an exponential-decay schedule.
- `num_epochs` — passes over the reference; `"auto"` scales the count with reference size. A few-hundred-cell reference needs several hundred epochs to train at all — at the inherited fixed 50 it lands near chance, silently. References of 5,000 cells and up get 50, unchanged. Pass an int to pin it.
- `batch_size` — minibatch size; `"auto"` scales it with reference size. Atlas-scale references get a proportionally bigger batch, which is faster and marginally more accurate; references under ~2k cells get a smaller one (down to 32), so each epoch is more than a handful of updates. Between those, 128.
- `use_raw` — `"auto"`, `True` or `False`. `"auto"` uses `adata.raw` counts when present.
- `max_cells_per_label` — balance and cap the training set to this many cells per class. Recommended for large atlases: it bounds memory and speeds up training.
- `seed` — seed for weight initialisation and minibatch shuffling.
- `print_cost` — print per-epoch training cost.
- `standardize` — z-score each selected gene using the reference's frozen mean and standard deviation, and apply the same transform to every query. Opt-in: it improves one-stage accuracy but shifts the softmax calibration that the refine and abstain thresholds are tuned against.

#### `predict`

```python
predict(adata, reference_model, output_label_name="celltype", outpath=None,
        output_h5ad=False, use_raw="auto", chunk_size=50000)
```

Annotate `adata` with a trained [`ReferenceModel`](#referencemodel), no retraining. Adds `obs[output_label_name]` and `obs[output_label_name + "_probability"]`, index-aligned, and returns `adata`.

- `adata` — query `AnnData`.
- `reference_model` — a `ReferenceModel`.
- `output_label_name` — `obs` column to write labels into.
- `outpath` — directory to also write `predicted_probabilities.txt` and `<output_label_name>_predicted_label.txt` into. `None` writes nothing.
- `output_h5ad` — with `outpath` set, also write `predicted_label.h5ad`.
- `use_raw` — as in `train_reference`.
- `chunk_size` — cells per prediction chunk, which bounds peak memory on large queries.

#### `celltype_predict_actinn`

```python
celltype_predict_actinn(adata, train_data_path, outpath, train_label_name="celltype",
                        output_label_name="celltype", output_h5ad=False, use_raw="auto")
```

One-off train-and-predict in a single call, mirroring the original ACTINN API. Genes are filtered jointly on the combined reference and query matrix, as in the original implementation. Writes `..._predicted_probabilities.txt` and `..._predicted_label.txt` to `outpath`. Returns `(adata, parameters)`.

- `adata` — query `AnnData`.
- `train_data_path` — path to the labelled reference `.h5ad`.
- `outpath` — directory for the text outputs.
- `train_label_name`, `output_label_name`, `output_h5ad`, `use_raw` — as above.

#### `ReferenceModel`

```python
ReferenceModel(params, norm_genes, select_idx, classes, mu=None, sd=None)
```

A trained model plus everything needed to map a new query. Normally built by `train_reference` rather than constructed directly.

Attributes:

- `params` — trained network weights and biases, as numpy arrays.
- `norm_genes` — gene set over which library-size normalisation is computed.
- `select_idx` — positions within `norm_genes` of the genes fed to the network.
- `classes` — cell-type names, indexed by the network's output units.
- `mu`, `sd` — per-gene mean and standard deviation when the model was trained with `standardize=True`; `None` otherwise.

Methods:

- `predict_proba(adata, use_raw="auto", chunk_size=50000, gene_names=None)` — softmax probabilities of shape `(n_cells, n_types)`, computed in cell chunks. `gene_names` overrides the identifiers taken from `adata`.
- `predict_frame(adata, use_raw="auto", chunk_size=50000)` — predicted label and probability as a DataFrame indexed by `adata.obs_names`.
- `save(path)` — write weights to `{path}.npz` and metadata to `{path}.json`.
- `ReferenceModel.load(path)` — classmethod; load a model written by `save`.

### Two-stage references

#### `build_hierarchical_reference`

```python
build_hierarchical_reference(ref, label_key, embeddings=None, n_groups=8,
                             hierarchy=None, ontology_key=None, **train_kwargs)
```

Build a [`HierarchicalReferenceModel`](#hierarchicalreferencemodel): one coarse classifier over groups, plus one fine classifier per group. Everything runs on CPU; only the embedding may want a GPU.

- `ref` — labelled reference `AnnData`.
- `label_key` — column of `ref.obs` holding the fine labels.
- `embeddings` — per-cell embedding array aligned to `ref`, from which the hierarchy is discovered. Provide this or `hierarchy`.
- `n_groups` — number of coarse groups to discover when using `embeddings`.
- `hierarchy` — precomputed `{cell_type: group}` mapping, for when the embedding was computed on a QC-filtered copy whose cells no longer align 1:1 to `ref`. Any label in `ref` missing from it goes to a catch-all group.
- `ontology_key` — column of `ref.obs` holding a Cell Ontology term per cell, recorded on the model as `class_to_cl` for ontology-aware scoring.
- `**train_kwargs` — forwarded to `train_reference` (`max_cells_per_label`, `num_epochs`, `standardize`, …).

#### `discover_hierarchy`

```python
discover_hierarchy(embeddings, labels, n_groups=8)
```

Cluster cell-type centroids in embedding space into coarse groups. Returns `{cell_type: group_id}`.

- `embeddings` — array of shape `(n_cells, n_dim)`, for example scPRINT 256-d vectors.
- `labels` — array of shape `(n_cells,)` of fine cell-type labels.
- `n_groups` — number of coarse groups.

#### `annotate`

```python
annotate(query, model, output_label_name="celltype", **kwargs)
```

Annotate a query `AnnData` with either model type. Returns `query`, annotated in place.

- `query` — query `AnnData`.
- `model` — a `HierarchicalReferenceModel`, `RefinedReference` or `ReferenceModel`.
- `output_label_name` — base name of the `obs` columns written.
- `**kwargs` — forwarded to the model's `predict` (`use_raw`, `chunk_size`, `min_prob`, …).

#### `HierarchicalReferenceModel`

```python
HierarchicalReferenceModel(coarse, fine, type_to_group, classes, class_to_cl=None,
                           class_to_tissue=None, build_info=None)
```

A coarse `ReferenceModel` plus one fine `ReferenceModel` per coarse group. Normally built by `build_hierarchical_reference` or loaded by `bundled_reference`.

- `coarse` — the group-level `ReferenceModel`.
- `fine` — `{group_id: ReferenceModel}`.
- `type_to_group` — `{cell_type: group_id}`.
- `classes` — all fine cell-type names.
- `class_to_cl` — `{cell_type: Cell Ontology id}`, when recorded.
- `class_to_tissue` — `{cell_type: set of tissues}`, used by `refine_to_tissue`.
- `build_info` — provenance recorded at build time.

Attributes and methods:

- `organism` — property; the organism the reference was built from, or `None` if not recorded.
- `predict_frame(adata, use_raw="auto", chunk_size=50000, min_prob=None, allowed_groups=None, allowed_classes=None)` — coarse-predict, route each cell to its group's fine model, and return `(frame, None)`, where `frame` has `celltype`, `celltype_probability` and `coarse`, indexed by `adata.obs_names`.
- `predict(adata, output_label_name="celltype", use_raw="auto", chunk_size=50000, min_prob=None, allowed_groups=None, allowed_classes=None)` — the same, written into `adata.obs` in place as `<name>`, `<name>_probability` and `<name>_coarse`. Returns `adata`.
- `save(path)` — write a directory of model files plus a manifest.
- `HierarchicalReferenceModel.load(path)` — classmethod; load a directory written by `save`.

Shared prediction parameters:

- `min_prob` — abstain threshold in 0–1. Cells whose final fine-label probability falls below it are labelled `"unknown"`. `None` disables abstention.
- `allowed_groups`, `allowed_classes` — restrict prediction to a subset of groups or `{group_id: set(class)}`. Normally set for you by `refine_to_query` or `refine_to_tissue`.

### Pre-trained references

#### `bundled_reference`

```python
bundled_reference(name="panhuman_distill_v1", download=True, quiet=False)
```

Load a pre-trained `HierarchicalReferenceModel` by name. Resolves in order: shipped inside the package, then the download cache, then fetched from the reference archive and cached. Only the smallest reference ships in the wheel, since all of them together exceed PyPI's file-size limit.

- `name` — reference name; see `list_references`.
- `download` — when `False`, raise instead of fetching a reference that is not already local.
- `quiet` — suppress download progress output.

#### `list_references`

```python
list_references(include_available=True)
```

Every reference you can load. Returns a list of dicts with `name`, `organism`, `n_classes`, `n_groups`, `size_mb`, `built`, `source` and `status` — `"installed"`, `"cached"` or `"available"`, the last meaning it downloads on first use.

- `include_available` — when `False`, list only references already installed or cached, without describing the ones that would need a download.

```python
for r in aj.list_references():
    print(r["name"], r["organism"], r["n_classes"], r["status"])
```

#### `default_reference_for`

```python
default_reference_for(organism)
```

Name of the reference to use for `organism`, or `None` if none is bundled.

- `organism` — organism name, for example `"human"`.

#### `infer_organism`

```python
infer_organism(var_names, sample=200)
```

Best-effort organism from Ensembl gene ids. Returns `None` when it cannot be determined.

- `var_names` — gene identifiers, normally `adata.var_names`.
- `sample` — how many identifiers to inspect.

### Query-adaptive refinement

#### `refine_to_query`

```python
refine_to_query(model, adata, tissue=None, use_raw="auto", chunk_size=50000,
                min_mass=1.0, min_top1=1, top1_conf=0.3)
```

Mask a broad reference down to the classes evidenced in `adata`, with no retraining. Returns a [`RefinedReference`](#refinedreference). This is a pruning pass rather than an accuracy fix: it reliably protects types that are genuinely present and never made accuracy worse in testing, but it rarely makes accuracy meaningfully better, because the classes that cause confusion are the model's genuinely confusable siblings of real types and carry the same confidence signature as real rare types.

- `model` — a `HierarchicalReferenceModel`.
- `adata` — the query the reference is being narrowed to.
- `tissue` — optionally also restrict to types plausible in a tissue: a name or list, or `"auto"` to read `adata.obs["tissue"]` or `obs["tissue_general"]`. The two filters compose — a class must be both evidenced and tissue-plausible.
- `use_raw`, `chunk_size` — as elsewhere.
- `min_mass`, `min_top1`, `top1_conf` — evidence thresholds, passed through to `detect_present_classes`. Defaults favour recall of rare true types.

```python
refined = aj.refine_to_query(model, adata)
adata = refined.predict(adata)
refined.evidence.sort_values("mass", ascending=False).head()
```

#### `refine_to_tissue`

```python
refine_to_tissue(model, tissue=None, adata=None)
```

Restrict a broad reference to the cell types the census records in a tissue, keeping pan-tissue types (immune, endothelial, stromal) available everywhere so that liver-resident T cells and macrophages are unaffected. No ground truth, no retraining. Returns a [`RefinedReference`](#refinedreference).

- `model` — a `HierarchicalReferenceModel`.
- `tissue` — a name or list from the census `tissue_general` vocabulary (`"liver"`, `"lung"`, `"blood"`, `"heart"`, …); common synonyms such as `"PBMC"` → blood are recognised.
- `adata` — when `tissue` is `None`, read the tissue from `adata.obs["tissue"]` or `obs["tissue_general"]`.

#### `detect_present_classes`

```python
detect_present_classes(model, adata, use_raw="auto", chunk_size=50000,
                       min_mass=1.0, min_top1=1, top1_conf=0.3)
```

Estimate which of a broad model's classes are actually evidenced in `adata`, using only the model's own unmasked predictions. A class is kept if either its probability summed over every query cell reaches `min_mass`, which catches real types that are consistently a plausible runner-up without ever winning, or it wins outright for at least `min_top1` cells at probability `top1_conf` or above, which catches rare types that are unambiguous for a handful of cells. Coarse groups left with no surviving fine classes are dropped.

Returns `(allowed_groups, allowed_classes, evidence)`: a set of group ids, a `{group_id: set(class)}` mapping, and a DataFrame with one row per group and class carrying `mass`, `top1_count`, `max_prob` and `kept` for inspection and tuning.

- `model`, `adata`, `use_raw`, `chunk_size` — as in `refine_to_query`.
- `min_mass` — summed-probability floor for the diffuse-evidence criterion.
- `min_top1` — how many cells a class must win outright for the rare-but-clear criterion.
- `top1_conf` — probability a win must reach to count towards `min_top1`.

#### `RefinedReference`

```python
RefinedReference(model, allowed_groups, allowed_classes, evidence)
```

A view over a `HierarchicalReferenceModel` restricted to a subset of classes. It wraps the same trained weights, so no retraining and no reference data are needed. Built by `refine_to_query` or `refine_to_tissue` rather than directly.

- `evidence` — attribute; the DataFrame from `detect_present_classes`, or `None` for a tissue-only refinement.
- `predict_frame(adata, **kwargs)` and `predict(adata, output_label_name="celltype", **kwargs)` — as on `HierarchicalReferenceModel`, with the mask applied.

### Novel cell types

#### `detect_novel_celltypes`

```python
detect_novel_celltypes(model, adata, cluster_key=None, min_prob=0.5, min_cells=10,
                       min_frac=0.5, use_raw="auto", chunk_size=50000,
                       resolution=1.0, n_markers=10, output_key="novel")
```

Flag coherent groups of cells the reference cannot confidently explain. Clusters the query, scores each cell's best reference probability, and marks a cluster as a candidate novel cell type when it is both large enough and mostly low-confidence.

Returns `(evidence, markers)`: a DataFrame with one row per cluster carrying `n_cells`, `frac_low_conf`, `median_conf`, `nearest_label` (what the reference would have called it) and `novel`, most-novel first; and `{novel_label: [gene, ...]}` for the flagged clusters. Also writes `obs[output_key]` and `obs[output_key + "_confidence"]`.

- `model` — a `HierarchicalReferenceModel` or `ReferenceModel`.
- `adata` — query `AnnData`.
- `cluster_key` — `obs` column of an existing clustering. When `None`, a Leiden clustering is computed, which needs `leidenalg` and `igraph`.
- `min_prob` — a cell counts as low-confidence when its best label probability falls below this.
- `min_cells` — size floor separating a real novel population from noise and doublets. Lower it to surface rarer candidates at the cost of more false alarms.
- `min_frac` — fraction of a cluster that must be low-confidence for it to be flagged.
- `use_raw`, `chunk_size` — as elsewhere.
- `resolution` — Leiden resolution, when clustering here.
- `n_markers` — top marker genes returned per novel cluster.
- `output_key` — `obs` column to write novel labels into.

```python
evidence, markers = aj.detect_novel_celltypes(model, adata, min_cells=10)
evidence[evidence.novel]
markers["novel_1"]
```

### Submodules

- `actinn_predict` — flat reference mapping: `train_reference`, `predict`, `celltype_predict_actinn`, `ReferenceModel`. Preprocessing never densifies the full matrix; gene matching and intersection happen on the sparse matrix and only the shared-gene submatrix is materialised.
- `hierarchy` — two-stage annotation, pre-trained reference loading and query-adaptive refinement.
- `actinn_utils` — the JAX model itself: initialisation, forward pass, loss and the JIT-compiled training step, in `(n_cells, n_genes)` orientation.

## How it works

- **Sparse-aware preprocessing** — gene-name matching and intersection happen on
  the sparse matrix; only the shared-gene submatrix is densified.
- **Normalization** follows ACTINN: per-cell library-size normalize to 1e4,
  `log2(x+1)`, then expr- and CV-percentile gene filtering. Optional per-gene
  standardization (`standardize=True`) aligns the query into the reference's
  feature space for an accuracy boost.
- **Model** — a 4-layer MLP (100→50→25→n_types), Glorot init, Adam with
  exponential-decay schedule, trained with a JIT-compiled epoch step.
- **Schedule scaled to the reference** — batch size and epoch count both default
  to `"auto"`. Above 5,000 cells this is exactly ACTINN's inherited schedule (50
  epochs; batch 128 up to 12.8k cells, then n/100 capped at 1024). Below it, the
  batch shrinks and the epoch count rises, because the inherited fixed schedule
  gives a 437-cell reference only ~170 gradient steps in total and returns
  near-chance labels for them. Training that pins too few total steps warns.
- **Cached `ReferenceModel`** stores the trained weights, gene set, and label
  mapping so queries are projected onto the reference's fixed gene space.

## Tests & benchmarks

```bash
uv pip install -e ".[test]"         # or: pip install -e ".[test]"
pytest                              # synthetic-data unit tests
python benchmark/benchmark.py       # timing on synthetic data
```

## License

MIT. See [LICENSE](https://github.com/iandriver/actinn-jax/blob/main/LICENSE).
