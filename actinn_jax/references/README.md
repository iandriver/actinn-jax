# Bundled references

Pre-trained `HierarchicalReferenceModel`s loadable with
`actinn_jax.bundled_reference(name)`. They annotate unknown human single-cell data on
CPU with no scPRINT/GPU at inference.

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

For a narrower, higher-accuracy reference on your own cell types, see
`examples/build_reference.py`.
