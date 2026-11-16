# actinn-jax

Label the cells in your single-cell dataset. Either use one of the pre-trained human
and mouse references that come with the package, or train your own on data you have
already labelled.

It runs on a laptop CPU. There is no GPU step, no TensorFlow, and no model to download
before you can start.

## Install

```bash
pip install actinn-jax
```

or from [bioconda](https://bioconda.github.io/recipes/actinn-jax/README.html):

```bash
conda install -c bioconda actinn-jax
```

That is everything you need. A pre-trained human reference, `panhuman_distill_v1`
(324 cell types, 16 MB), is included in the package, so the example below runs offline.

## Try it in one minute

Take pbmc3k, the 2,700-cell dataset from the Scanpy tutorial. Raw counts, no labels, no
clustering:

```python
import scanpy as sc, actinn_jax as aj

adata = sc.datasets.pbmc3k()                          # raw counts, no labels
adata = aj.annotate(adata, aj.bundled_reference())
```

Two lines, about 0.3 seconds on an M-series laptop. You now have a label and a confidence
score for every cell:

```
                            celltype  celltype_probability
AAACATACAACCAC-1    Naive CD8 T cell                  0.45
AAACATTGAGCTAC-1       Memory B cell                  0.99
AAACATTGATCAGC-1   Memory CD4 T cell                  0.61
AAACCGTGCTTCCG-1  CD14/CD16 monocyte                  0.83
AAACCGTGTATGCG-1        CD16 NK cell                  0.90
```

The Scanpy tutorial gets to eight cell types on this dataset by clustering and then
reading marker genes by hand. Here is how those eight line up against what actinn-jax
called the same 2,638 cells:

| Tutorial label | Cells | actinn-jax called them |
|---|---|---|
| CD4 T cells | 1144 | Naive CD4 T 63%, Memory CD4 T 16% |
| CD14+ Monocytes | 480 | CD14 monocyte 49%, CD14/CD16 monocyte 46% |
| B cells | 342 | Naive B 57%, Memory B 41% |
| CD8 T cells | 316 | GZMK CD8 T 41%, GZMB CD8 T 22%, MAIT 15% |
| NK cells | 154 | CD16 NK 87%, CD56 NK 6% |
| FCGR3A+ Monocytes | 150 | CD16 monocyte 78% |
| Dendritic cells | 37 | cDC2 62%, pDC 11%, cDC1 8% |
| Megakaryocytes | 15 | Platelet 87% |

Same groups, no marker genes read, and finer splits where the reference knows them:
naive against memory B cells, and three separate CD8 T subsets.

## Gene identifiers

The pbmc3k example above works even though its `var_names` are gene symbols and the
reference is keyed by Ensembl IDs. When the names do not overlap, actinn-jax looks
through the columns of `adata.var` for one that does, uses it, and prints which column
it picked.

If nothing matches, it raises an error. That matters, because the alternative is what
you get from most tools: one label repeated for all of your cells, with no warning.

To choose the identifier yourself:

```python
adata.var_names = adata.var["gene_ids"].astype(str).values
adata.var_names_make_unique()
```

## Raw counts

actinn-jax expects raw integer counts, because it does its own normalisation (counts per
10,000, then log2). For CELLxGENE-style objects where `.X` holds normalised values and
the counts are in `.raw`, it finds the counts on its own.

If you hand it values that scanpy has already normalised, it warns:

```
UserWarning: actinn-jax: adata.X does not look like raw counts (it holds
non-integer values). ACTINN applies its own CP10k+log2 normalization, so
already-normalized input is normalized twice and the labels will shift
without erroring.
```

That warning is worth acting on. Normalising twice does not fail and does not produce
obvious nonsense; it produces confident labels that differ. On pbmc3k, running
`sc.pp.normalize_total` and `sc.pp.log1p` before annotating changes the label for 7.6% of
cells, and the changes land on real distinctions: 67 cells move between naive and memory
CD4 T, 25 between CD14 and CD14/CD16 monocyte.

So if you are partway through a scanpy workflow, keep the counts somewhere:

```python
adata.layers["counts"] = adata.X.copy()     # before you normalise
...
adata.X = adata.layers["counts"]            # then annotate
adata = aj.annotate(adata, model)
```

## Cells the reference cannot explain

By default every cell gets the closest available label, even when the reference has
nothing suitable. Set `min_prob` and low-confidence cells are labelled `"unknown"`
instead:

```python
adata = aj.annotate(adata, model, min_prob=0.5)
```

Held-out calibration on the 798-type census reference, with 10% of cell types withheld
entirely so they cannot be right:

| `min_prob` | Accuracy on kept cells | Cells kept | Withheld types caught |
|---:|---:|---:|---:|
| 0 (off) | 0.55 | 100% | 0% |
| 0.5 | 0.80 | 46% | 75% |
| 0.7 | 0.86 | 30% | 88% |

Accuracy is lower here than you would see from a reference built for one tissue. Picking
between 800 cell types is a harder question than picking between 40, and many of the 800
are near-duplicates such as cortical neuron layers. If you care about accuracy more than
breadth, train a focused reference (below). Each reference reports its own calibration in
`build_info.json`; `broad_mouse_v1` reaches 0.88 accuracy while keeping 82% of cells.

## Finding cell types the reference has never seen

A genuinely new cell type does not appear as scattered uncertain cells. It appears as a
coherent group that the reference cannot explain. `detect_novel_celltypes` looks for
clusters that are both large enough to be real and mostly low-confidence:

```python
evidence, markers = aj.detect_novel_celltypes(
    model, adata,
    cluster_key="leiden",   # your clustering; omit and it will cluster for you
    min_cells=10,           # smallest population worth calling novel
)
evidence[evidence.novel]    # candidates, most suspect first
markers["novel_1"]          # what makes it distinct
```

We tested this by hiding a known answer. In the Krasnow et al. (2020) human lung atlas we
trained a reference with the pulmonary ionocyte removed, a cell type that was itself a
landmark discovery in 2018, then asked the method to find what was missing. Out of 57
populations it flagged one: the 22 ionocytes, at 86% low confidence. The reference would
otherwise have called them B cells. The marker genes it recovered were led by ASCL3, the
ionocyte master transcription factor, and ATP6V0B, a subunit of the proton pump the cell
type is named after.

Raising `min_cells` above 22 makes the flag go away, which is the behaviour you want from
a size threshold. The walkthrough is in
[examples/novel_celltypes.ipynb](examples/novel_celltypes.ipynb).

## Narrowing a broad reference to your data

A pan-human reference knows about cell types that cannot be in your sample. Two functions
remove them without retraining and without any labels from you.

`refine_to_tissue` keeps only the cell types the CELLxGENE census records in a given
tissue, while leaving immune, endothelial and stromal types available everywhere, so
liver-resident T cells and macrophages survive:

```python
refined = aj.refine_to_tissue(model, tissue="liver")   # or ["liver", "blood"]
adata = aj.annotate(adata, refined)

refined = aj.refine_to_tissue(model, adata=adata)      # or read it from adata.obs
```

On a real liver query this cuts the label set roughly in half. It removes cardiac muscle,
alveolar fibroblast, colonocyte and adrenal cortex, while hepatocyte, LSEC and stellate
counts stay where they were. Common synonyms like PBMC for blood are understood, and a
tissue the reference does not know simply applies no filter.

`refine_to_query` keeps only the classes your query's own predictions support:

```python
refined = aj.refine_to_query(model, adata, tissue="liver")   # tissue is optional
adata = aj.annotate(adata, refined)
refined.evidence.sort_values("mass", ascending=False).head()
```

Be clear about what this buys you. Measured against ground truth
([REFINE.md](https://github.com/iandriver/actinn-jax-benchmark/blob/main/docs/REFINE.md)),
it is a safe pruning pass: it keeps the types that are genuinely present and never made
accuracy worse in our testing. It rarely makes accuracy better. The classes that cause
real confusion are the model's own confusable siblings of true types, and they look
exactly like rare-but-real types to any threshold built from the classifier's confidence.
Closing that gap takes a narrower reference, not a better filter.

## Training your own reference

A reference built for your tissue beats a broad one by a wide margin. On a held-out
study, a 48-type liver reference scores 0.72 exact and 0.86 ontology concordance, against
0.23 and 0.58 for the broad 798-type reference on the same cells.

```python
model = aj.train_reference("reference.h5ad", train_label_name="cell_type")
model.save("my_reference")

model = aj.ReferenceModel.load("my_reference")
adata, _ = aj.predict(adata, model, output_label_name="celltype")
```

For an atlas-sized reference, cap the cells per class to bound memory and speed up
training:

```python
model = aj.train_reference("big_atlas.h5ad", train_label_name="cell_type",
                           max_cells_per_label=500)
adata, _ = aj.predict(adata, model, chunk_size=50000)
```

Setting `standardize=True` z-scores each gene using the reference's mean and standard
deviation and applies the same transform to every query. Across the Open Problems
`label_projection` datasets it adds 0.3 points of accuracy and 1.1 points of macro-F1,
with the biggest gains where reference and query differ most. It is off by default because
it shifts the confidence calibration that `min_prob` and the refine thresholds are tuned
against, so use it on its own or re-tune those first.

To build a two-stage reference, where a coarse classifier routes each cell to a
specialist, you need cell embeddings. This is the one step that wants a GPU, and it
happens once:

```python
from actinn_jax.embed import scprint_embed
emb = scprint_embed(ref_adata)
model = aj.build_hierarchical_reference(ref_adata, "cell_type", emb)
model.save("my_reference")
```

Annotating with that model is CPU-only, like everything else.

## Which references are available

```python
for r in aj.list_references():
    print(r["name"], r["organism"], r["n_classes"], r["status"])
```

`panhuman_distill_v1` is included in the package. The census-scale references are tens of
MB each and download once from Zenodo
([10.5281/zenodo.21688151](https://doi.org/10.5281/zenodo.21688151)) into
`~/.cache/actinn-jax/`, then stay there. Each archive is checksum-verified before it is
unpacked. `liver_hlica_v2` is a 48-type liver reference built from
[HLiCA](https://doi.org/10.64898/2026.06.30.735539), including hepatocyte and endothelial
zonation; cite that paper if you use it.

Loading a reference for the wrong organism raises an error rather than returning
nonsense.

To download references ahead of a workshop or a flight:

```bash
actinn-jax fetch                  # all of them
actinn-jax fetch broad_mouse_v1   # or one
actinn-jax references             # what you have
actinn-jax cache --clear          # get the disk space back
```

Set `ACTINN_JAX_CACHE` to move the cache, which helps on a shared cluster, and
`ACTINN_JAX_REFERENCE_URL` to serve the archives from your own mirror.

## From the command line

```bash
actinn-jax annotate my_data.h5ad --min-prob 0.5 -o annotations.csv
```

This reads the organism from the gene IDs and picks the matching reference.

## Installing from source

```bash
git clone https://github.com/iandriver/actinn-jax.git
cd actinn-jax
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
```

[uv](https://docs.astral.sh/uv/) resolves the scientific stack faster and will fetch a
suitable Python for you:

```bash
uv venv --python 3.12
uv pip install -e .
```

You need Python 3.10 to 3.12. Newer versions may not have JAX or scanpy wheels yet.

The example notebooks need Jupyter, matplotlib and a few extras:

```bash
pip install -e ".[notebooks]"
jupyter lab examples/
```

Each notebook has a `QUERY = '...'` line near the top. Point it at your own `.h5ad` of
raw counts.

On Apple Silicon there is an experimental GPU backend (`pip install -e ".[metal]"`), but
the model is small enough that it is not faster than CPU.

## Examples

- [quickstart_annotate.py](examples/quickstart_annotate.py), the shortest working script
- [annotate_with_timing.ipynb](examples/annotate_with_timing.ipynb), a 65k-cell lung atlas
  end to end, with throughput and an abstain sweep
- [novel_celltypes.ipynb](examples/novel_celltypes.ipynb), the ionocyte experiment above
- [liver_zonation.ipynb](examples/liver_zonation.ipynb), mapping hepatocytes onto the
  portal to central axis
- [build_reference.py](examples/build_reference.py), building your own

## API reference

Every public name, with its parameters. The examples assume `import actinn_jax as aj`.

- [Training and prediction](#training-and-prediction): `train_reference`, `predict`, `celltype_predict_actinn`, `ReferenceModel`
- [Two-stage references](#two-stage-references): `build_hierarchical_reference`, `discover_hierarchy`, `annotate`, `HierarchicalReferenceModel`
- [Pre-trained references](#pre-trained-references): `bundled_reference`, `list_references`, `default_reference_for`, `infer_organism`
- [Query-adaptive refinement](#query-adaptive-refinement): `refine_to_query`, `refine_to_tissue`, `detect_present_classes`, `RefinedReference`
- [Novel cell types](#novel-cell-types): `detect_novel_celltypes`
- [Submodules](#submodules): `actinn_predict`, `hierarchy`, `actinn_utils`

### Training and prediction

#### `train_reference`

```python
train_reference(train_data, train_label_name="celltype", learning_rate=0.0001,
                num_epochs=50, batch_size="auto", use_raw="auto",
                max_cells_per_label=None, seed=3, print_cost=True,
                standardize=False)
```

Train a classifier on a labelled reference. Returns a [`ReferenceModel`](#referencemodel).

- `train_data`: `.h5ad` path or `AnnData` holding raw counts (in `.raw` or `.X`) and cell-type labels in `obs[train_label_name]`.
- `train_label_name`: column of `obs` holding the labels.
- `learning_rate`: Adam learning rate, used with an exponential-decay schedule.
- `num_epochs`: passes over the reference.
- `batch_size`: minibatch size; `"auto"` scales it with reference size. References under ~12.8k cells keep a batch of 128; larger ones get a proportionally bigger batch, which is faster and marginally more accurate at atlas scale.
- `use_raw`: `"auto"`, `True` or `False`. `"auto"` uses `adata.raw` counts when present.
- `max_cells_per_label`: balance and cap the training set to this many cells per class. Recommended for large atlases: it bounds memory and speeds up training.
- `seed`: seed for weight initialisation and minibatch shuffling.
- `print_cost`: print per-epoch training cost.
- `standardize`: z-score each selected gene using the reference's frozen mean and standard deviation, and apply the same transform to every query. Opt-in: it improves one-stage accuracy but shifts the softmax calibration that the refine and abstain thresholds are tuned against.

#### `predict`

```python
predict(adata, reference_model, output_label_name="celltype", outpath=None,
        output_h5ad=False, use_raw="auto", chunk_size=50000)
```

Annotate `adata` with a trained [`ReferenceModel`](#referencemodel), no retraining. Adds `obs[output_label_name]` and `obs[output_label_name + "_probability"]`, index-aligned, and returns `adata`.

- `adata`: query `AnnData`.
- `reference_model`: a `ReferenceModel`.
- `output_label_name`: `obs` column to write labels into.
- `outpath`: directory to also write `predicted_probabilities.txt` and `<output_label_name>_predicted_label.txt` into. `None` writes nothing.
- `output_h5ad`: with `outpath` set, also write `predicted_label.h5ad`.
- `use_raw`: as in `train_reference`.
- `chunk_size`: cells per prediction chunk, which bounds peak memory on large queries.

#### `celltype_predict_actinn`

```python
celltype_predict_actinn(adata, train_data_path, outpath, train_label_name="celltype",
                        output_label_name="celltype", output_h5ad=False, use_raw="auto")
```

One-off train-and-predict in a single call, mirroring the original ACTINN API. Genes are filtered jointly on the combined reference and query matrix, as in the original implementation. Writes `..._predicted_probabilities.txt` and `..._predicted_label.txt` to `outpath`. Returns `(adata, parameters)`.

- `adata`: query `AnnData`.
- `train_data_path`: path to the labelled reference `.h5ad`.
- `outpath`: directory for the text outputs.
- `train_label_name`, `output_label_name`, `output_h5ad`, `use_raw`, as above.

#### `ReferenceModel`

```python
ReferenceModel(params, norm_genes, select_idx, classes, mu=None, sd=None)
```

A trained model plus everything needed to map a new query. Normally built by `train_reference` rather than constructed directly.

Attributes:

- `params`: trained network weights and biases, as numpy arrays.
- `norm_genes`: gene set over which library-size normalisation is computed.
- `select_idx`: positions within `norm_genes` of the genes fed to the network.
- `classes`: cell-type names, indexed by the network's output units.
- `mu`, `sd`, per-gene mean and standard deviation when the model was trained with `standardize=True`; `None` otherwise.

Methods:

- `predict_proba(adata, use_raw="auto", chunk_size=50000, gene_names=None)`: softmax probabilities of shape `(n_cells, n_types)`, computed in cell chunks. `gene_names` overrides the identifiers taken from `adata`.
- `predict_frame(adata, use_raw="auto", chunk_size=50000)`: predicted label and probability as a DataFrame indexed by `adata.obs_names`.
- `save(path)`: write weights to `{path}.npz` and metadata to `{path}.json`.
- `ReferenceModel.load(path)`: classmethod; load a model written by `save`.

### Two-stage references

#### `build_hierarchical_reference`

```python
build_hierarchical_reference(ref, label_key, embeddings=None, n_groups=8,
                             hierarchy=None, ontology_key=None, **train_kwargs)
```

Build a [`HierarchicalReferenceModel`](#hierarchicalreferencemodel): one coarse classifier over groups, plus one fine classifier per group. Everything runs on CPU; only the embedding may want a GPU.

- `ref`: labelled reference `AnnData`.
- `label_key`: column of `ref.obs` holding the fine labels.
- `embeddings`: per-cell embedding array aligned to `ref`, from which the hierarchy is discovered. Provide this or `hierarchy`.
- `n_groups`: number of coarse groups to discover when using `embeddings`.
- `hierarchy`: precomputed `{cell_type: group}` mapping, for when the embedding was computed on a QC-filtered copy whose cells no longer align 1:1 to `ref`. Any label in `ref` missing from it goes to a catch-all group.
- `ontology_key`: column of `ref.obs` holding a Cell Ontology term per cell, recorded on the model as `class_to_cl` for ontology-aware scoring.
- `**train_kwargs`: forwarded to `train_reference` (`max_cells_per_label`, `num_epochs`, `standardize`, …).

#### `discover_hierarchy`

```python
discover_hierarchy(embeddings, labels, n_groups=8)
```

Cluster cell-type centroids in embedding space into coarse groups. Returns `{cell_type: group_id}`.

- `embeddings`: array of shape `(n_cells, n_dim)`, for example scPRINT 256-d vectors.
- `labels`: array of shape `(n_cells,)` of fine cell-type labels.
- `n_groups`: number of coarse groups.

#### `annotate`

```python
annotate(query, model, output_label_name="celltype", **kwargs)
```

Annotate a query `AnnData` with either model type. Returns `query`, annotated in place.

- `query`: query `AnnData`.
- `model`: a `HierarchicalReferenceModel`, `RefinedReference` or `ReferenceModel`.
- `output_label_name`: base name of the `obs` columns written.
- `**kwargs`: forwarded to the model's `predict` (`use_raw`, `chunk_size`, `min_prob`, …).

#### `HierarchicalReferenceModel`

```python
HierarchicalReferenceModel(coarse, fine, type_to_group, classes, class_to_cl=None,
                           class_to_tissue=None, build_info=None)
```

A coarse `ReferenceModel` plus one fine `ReferenceModel` per coarse group. Normally built by `build_hierarchical_reference` or loaded by `bundled_reference`.

- `coarse`: the group-level `ReferenceModel`.
- `fine`: `{group_id: ReferenceModel}`.
- `type_to_group`: `{cell_type: group_id}`.
- `classes`: all fine cell-type names.
- `class_to_cl`: `{cell_type: Cell Ontology id}`, when recorded.
- `class_to_tissue`: `{cell_type: set of tissues}`, used by `refine_to_tissue`.
- `build_info`: provenance recorded at build time.

Attributes and methods:

- `organism`: property; the organism the reference was built from, or `None` if not recorded.
- `predict_frame(adata, use_raw="auto", chunk_size=50000, min_prob=None, allowed_groups=None, allowed_classes=None)`: coarse-predict, route each cell to its group's fine model, and return `(frame, None)`, where `frame` has `celltype`, `celltype_probability` and `coarse`, indexed by `adata.obs_names`.
- `predict(adata, output_label_name="celltype", use_raw="auto", chunk_size=50000, min_prob=None, allowed_groups=None, allowed_classes=None)`: the same, written into `adata.obs` in place as `<name>`, `<name>_probability` and `<name>_coarse`. Returns `adata`.
- `save(path)`: write a directory of model files plus a manifest.
- `HierarchicalReferenceModel.load(path)`: classmethod; load a directory written by `save`.

Shared prediction parameters:

- `min_prob`: abstain threshold between 0 and 1. Cells whose final fine-label probability falls below it are labelled `"unknown"`. `None` disables abstention.
- `allowed_groups`, `allowed_classes`, restrict prediction to a subset of groups or `{group_id: set(class)}`. Normally set for you by `refine_to_query` or `refine_to_tissue`.

### Pre-trained references

#### `bundled_reference`

```python
bundled_reference(name="panhuman_distill_v1", download=True, quiet=False)
```

Load a pre-trained `HierarchicalReferenceModel` by name. Resolves in order: shipped inside the package, then the download cache, then fetched from the reference archive and cached. Only the smallest reference ships in the wheel, since all of them together exceed PyPI's file-size limit.

- `name`: reference name; see `list_references`.
- `download`: when `False`, raise instead of fetching a reference that is not already local.
- `quiet`: suppress download progress output.

#### `list_references`

```python
list_references(include_available=True)
```

Every reference you can load. Returns a list of dicts with `name`, `organism`, `n_classes`, `n_groups`, `size_mb`, `built`, `source` and `status`, `"installed"`, `"cached"` or `"available"`, the last meaning it downloads on first use.

- `include_available`: when `False`, list only references already installed or cached, without describing the ones that would need a download.

```python
for r in aj.list_references():
    print(r["name"], r["organism"], r["n_classes"], r["status"])
```

#### `default_reference_for`

```python
default_reference_for(organism)
```

Name of the reference to use for `organism`, or `None` if none is bundled.

- `organism`: organism name, for example `"human"`.

#### `infer_organism`

```python
infer_organism(var_names, sample=200)
```

Best-effort organism from Ensembl gene ids. Returns `None` when it cannot be determined.

- `var_names`: gene identifiers, normally `adata.var_names`.
- `sample`: how many identifiers to inspect.

### Query-adaptive refinement

#### `refine_to_query`

```python
refine_to_query(model, adata, tissue=None, use_raw="auto", chunk_size=50000,
                min_mass=1.0, min_top1=1, top1_conf=0.3)
```

Mask a broad reference down to the classes evidenced in `adata`, with no retraining. Returns a [`RefinedReference`](#refinedreference). This is a pruning pass rather than an accuracy fix: it reliably protects types that are genuinely present and never made accuracy worse in testing, but it rarely makes accuracy meaningfully better, because the classes that cause confusion are the model's genuinely confusable siblings of real types and carry the same confidence signature as real rare types.

- `model`: a `HierarchicalReferenceModel`.
- `adata`: the query the reference is being narrowed to.
- `tissue`: optionally also restrict to types plausible in a tissue: a name or list, or `"auto"` to read `adata.obs["tissue"]` or `obs["tissue_general"]`. The two filters compose, a class must be both evidenced and tissue-plausible.
- `use_raw`, `chunk_size`, as elsewhere.
- `min_mass`, `min_top1`, `top1_conf`, evidence thresholds, passed through to `detect_present_classes`. Defaults favour recall of rare true types.

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

- `model`: a `HierarchicalReferenceModel`.
- `tissue`: a name or list from the census `tissue_general` vocabulary (`"liver"`, `"lung"`, `"blood"`, `"heart"`, …); common synonyms such as `"PBMC"` to blood are recognised.
- `adata`: when `tissue` is `None`, read the tissue from `adata.obs["tissue"]` or `obs["tissue_general"]`.

#### `detect_present_classes`

```python
detect_present_classes(model, adata, use_raw="auto", chunk_size=50000,
                       min_mass=1.0, min_top1=1, top1_conf=0.3)
```

Estimate which of a broad model's classes are actually evidenced in `adata`, using only the model's own unmasked predictions. A class is kept if either its probability summed over every query cell reaches `min_mass`, which catches real types that are consistently a plausible runner-up without ever winning, or it wins outright for at least `min_top1` cells at probability `top1_conf` or above, which catches rare types that are unambiguous for a handful of cells. Coarse groups left with no surviving fine classes are dropped.

Returns `(allowed_groups, allowed_classes, evidence)`: a set of group ids, a `{group_id: set(class)}` mapping, and a DataFrame with one row per group and class carrying `mass`, `top1_count`, `max_prob` and `kept` for inspection and tuning.

- `model`, `adata`, `use_raw`, `chunk_size`, as in `refine_to_query`.
- `min_mass`: summed-probability floor for the diffuse-evidence criterion.
- `min_top1`: how many cells a class must win outright for the rare-but-clear criterion.
- `top1_conf`: probability a win must reach to count towards `min_top1`.

#### `RefinedReference`

```python
RefinedReference(model, allowed_groups, allowed_classes, evidence)
```

A view over a `HierarchicalReferenceModel` restricted to a subset of classes. It wraps the same trained weights, so no retraining and no reference data are needed. Built by `refine_to_query` or `refine_to_tissue` rather than directly.

- `evidence`: attribute; the DataFrame from `detect_present_classes`, or `None` for a tissue-only refinement.
- `predict_frame(adata, **kwargs)` and `predict(adata, output_label_name="celltype", **kwargs)`, as on `HierarchicalReferenceModel`, with the mask applied.

### Novel cell types

#### `detect_novel_celltypes`

```python
detect_novel_celltypes(model, adata, cluster_key=None, min_prob=0.5, min_cells=10,
                       min_frac=0.5, use_raw="auto", chunk_size=50000,
                       resolution=1.0, n_markers=10, output_key="novel")
```

Flag coherent groups of cells the reference cannot confidently explain. Clusters the query, scores each cell's best reference probability, and marks a cluster as a candidate novel cell type when it is both large enough and mostly low-confidence.

Returns `(evidence, markers)`: a DataFrame with one row per cluster carrying `n_cells`, `frac_low_conf`, `median_conf`, `nearest_label` (what the reference would have called it) and `novel`, most-novel first; and `{novel_label: [gene, ...]}` for the flagged clusters. Also writes `obs[output_key]` and `obs[output_key + "_confidence"]`.

- `model`: a `HierarchicalReferenceModel` or `ReferenceModel`.
- `adata`: query `AnnData`.
- `cluster_key`: `obs` column of an existing clustering. When `None`, a Leiden clustering is computed, which needs `leidenalg` and `igraph`.
- `min_prob`: a cell counts as low-confidence when its best label probability falls below this.
- `min_cells`: size floor separating a real novel population from noise and doublets. Lower it to surface rarer candidates at the cost of more false alarms.
- `min_frac`: fraction of a cluster that must be low-confidence for it to be flagged.
- `use_raw`, `chunk_size`, as elsewhere.
- `resolution`: Leiden resolution, when clustering here.
- `n_markers`: top marker genes returned per novel cluster.
- `output_key`: `obs` column to write novel labels into.

```python
evidence, markers = aj.detect_novel_celltypes(model, adata, min_cells=10)
evidence[evidence.novel]
markers["novel_1"]
```

### Submodules

- `actinn_predict`: flat reference mapping: `train_reference`, `predict`, `celltype_predict_actinn`, `ReferenceModel`. Preprocessing never densifies the full matrix; gene matching and intersection happen on the sparse matrix and only the shared-gene submatrix is materialised.
- `hierarchy`: two-stage annotation, pre-trained reference loading and query-adaptive refinement.
- `actinn_utils`: the JAX model itself: initialisation, forward pass, loss and the JIT-compiled training step, in `(n_cells, n_genes)` orientation.

## How it works

The classifier is a four-layer neural network (100, 50, 25 hidden units, then one output
per cell type) trained with Adam. Gene matching and intersection happen on the sparse
matrix, and only the shared genes are made dense, so a large query never fully expands in
memory. Counts are normalised per cell to 10,000 and log-transformed, then filtered by
expression and coefficient of variation, following the original ACTINN.

A trained model stores its weights, its gene set and its label mapping, so every query is
projected onto the same fixed gene space the reference was trained on.

A two-stage model works out a coarse-to-fine grouping of cell types once, offline, using
embeddings from a foundation model. It then trains a coarse classifier plus one
specialist per group. Annotating new data uses only the small classifiers, on CPU.

## Where the numbers come from

Accuracy, speed and memory comparisons against other methods live in the companion
[actinn-jax-benchmark](https://github.com/iandriver/actinn-jax-benchmark) repository.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Credit

This package implements ACTINN. The method is not ours; please cite the original paper:

> Feiyang Ma, Matteo Pellegrini. ACTINN: automated identification of cell types in single
> cell RNA sequencing. *Bioinformatics*, 2020.
> https://doi.org/10.1093/bioinformatics/btz592

The original implementation, in TensorFlow 1.x, is at
https://github.com/mafeiyang/ACTINN. This version replaces it with a JIT-compiled JAX
network and sparse-aware preprocessing, adds a saved reference model you train once and
reuse, and scales to atlas-sized data.

## License

MIT. See [LICENSE](LICENSE).
