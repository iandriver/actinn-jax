# Bundled references

Pre-trained `HierarchicalReferenceModel`s loadable with
`actinn_jax.bundled_reference(name)`. They annotate unknown human single-cell data on
CPU with no scPRINT/GPU at inference.

## `panhuman_distill_v1`

A 324-class reference **distilled from Pan-human Azimuth**, which supplies both the labels
and the coarse→fine hierarchy. It speaks a harmonized vocabulary with a one-to-one Cell
Ontology crosswalk rather than raw census strings, and it is the more accurate of the two
broad references here.

- **Source:** 85,256 cells — krasnow lung, HLiCA liver, blood+gut, and a CELLxGENE census
  pull (2025-11-08; 51,346 cells / 867 types / 376 tissues). Pan-human Azimuth labels every
  cell; the atlases' own labels are used only for scoring, never for training.
- **Build:** no GPU and no labeled input — the teacher's own broad level is the hierarchy,
  so the scPRINT embedding step is not needed. Under 10 minutes of CPU after the data pull.
- **Size:** ~16 MB.
- **On a withheld HLiCA liver study** (3,396 cells, never in the corpus):

  | model | classes | ontology concordance | cells/s |
  |---|---:|---:|---:|
  | `broad_human_v1` | 798 | 0.338 | 2,962 |
  | **`panhuman_distill_v1`** | 324 | **0.406** | **8,937–10,021** |
  | Pan-human Azimuth itself | 47 | 0.380 | ~1,000–1,500 |

- **Withholding a whole atlas** costs little: the student tracks its teacher to within
  1.5 points on lung (0.695 vs 0.710) and 3.0 on liver (0.481 vs 0.511).

### Attribution — required

The teacher's weights are **CC BY 4.0**, so credit is a licence condition. Work using this
reference must cite:

> Sarkar, Li, Molla, … Satija. *Organism-scale annotation with Pan-human Azimuth.* bioRxiv
> 2026. [doi:10.64898/2026.07.16.738997](https://doi.org/10.64898/2026.07.16.738997).
> Model weights © the authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
> via [`panhumanpy`](https://github.com/satijalab/panhumanpy) (MIT) and
> [Zenodo](https://doi.org/10.5281/zenodo.20401417).

The same notice travels inside the model at `panhuman_distill_v1/build_info.json`.

## `broad_mouse_v1`

Census-wide **mouse** reference. Pan-human Azimuth is human-only and scPRINT's mouse support
is untested here, so this one takes its coarse→fine hierarchy from **Cell Ontology lineage**
instead of a foundation-model embedding — free, deterministic, and species-independent.

- **Source:** CELLxGENE census (2025-11-08), primary mouse cells, ≤60 per type →
  **27,026 cells, 453 cell types, 85 tissues** across 47 datasets. Two datasets were
  excluded from the reference entirely and used as the test set.
- **Build:** 17 s on CPU. No GPU, no foundation model.
- **Size:** ~38 MB.
- **Held out** (12,646 cells from the two withheld datasets, 137 truth types, 41 tissues):
  ontology concordance **0.638**, rising to **0.718** on the 71% of cells kept at
  `min_prob=0.5`; 9,712 cells/s.
- **Abstain calibration** (45 types withheld as out-of-distribution):

  | `min_prob` | accuracy (kept) | coverage | OOD flagged |
  |-----------:|----------------:|---------:|------------:|
  | 0.0 | 0.79 | 100% | 0% |
  | 0.5 | 0.88 | 82% | 49% |
  | 0.7 | 0.91 | 68% | 69% |

Details and the hierarchy ablation that justifies the CL route: `docs/PAN_MOUSE.md` in the
companion repo.

## `broad_human_v1`

Census-wide human reference — the breadth scPRINT itself was trained on.

- **Source:** CELLxGENE census (`stable`, 2025-11-08), all primary human cells
  (96.6M cells / 872 cell types). Stratified sample of ≤40 cells per cell type across
  440 datasets → **26,973 cells, 798 cell types, 314 tissues**.
- **Build:** scPRINT (`medium-v1.5`) embeds the reference once offline; cell-type
  centroids are clustered into **28 coarse groups** (Ward linkage); a coarse actinn-jax
  classifier + one fine classifier per group are trained on a 4,000-gene HVG panel.
- **Size:** ~50 MB (float32 weights + gene lists).
- **Held-out calibration** (10% of types held out entirely as out-of-distribution):

  | `min_prob` | accuracy (kept) | coverage | OOD flagged |
  |-----------:|----------------:|---------:|------------:|
  | 0.0 | 0.55 | 100% | 0% |
  | 0.5 | 0.80 | 46% | 75% |
  | 0.7 | 0.86 | 30% | 88% |

  Use `annotate(adata, model, min_prob=0.5)` to abstain (`"unknown"`) on low-confidence
  / out-of-distribution cells rather than force-labeling them.

## Rebuilding

One command in the companion repo
[actinn-jax-benchmark](https://github.com/iandriver/actinn-jax-benchmark):

```bash
benchmark/explore/update_broad_reference.sh
```

It chains the three build stages (`fetch_census_wide.py` → `embed_broad.py` →
`build_census_model.py`) through one work directory, backs up the reference it is about to
replace, and then **scores the rebuilt model on a held-out atlas** — a rebuild can train
cleanly and still annotate worse. Full write-up, knobs, and the numbers a healthy rebuild
should land near: `docs/UPDATE_BROAD_REFERENCE.md`.

A rebuilt reference carries a `build_info.json` recording the census release it came from,
its sizes, and its calibration table. `stable` is a moving pointer, so pin
`CENSUS_VERSION=2025-11-08` to reproduce *this* reference.

`panhuman_distill_v1` is rebuilt by a different, cheaper route — no GPU, no labels — in the
same repo: `distill_dump.py` → `distill_train.py`, documented in `docs/PANHUMAN_DISTILL.md`.

For a narrower, higher-accuracy reference on your own cell types, see
`examples/build_reference.py`.
