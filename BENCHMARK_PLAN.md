# Benchmark plan: accuracy × runtime comparison of reference-based cell-type annotation

## Context & goal

Reference-based annotation (fit on a labeled reference, transfer labels to a query)
has many tools, but the literature splits the question: the most rigorous recent
accuracy benchmark ([Huang et al., *Brief. Bioinform.* 2024, bbae392](https://academic.oup.com/bib/article/25/5/bbae392/7730135))
reports **no runtime or memory**, while the canonical study that did
([Abdelaal et al., *Genome Biol.* 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1795-z))
is now dated and pre-foundation-model. This benchmark produces a **modern,
comprehensive, neutral accuracy × runtime × memory comparison** across classical
classifiers, deep reference-mapping, and foundation models, run on commodity
Apple-Silicon hardware (no CUDA), with `actinn-jax` included as one method among
many.

Decisions locked with the user:
- **Method tiers:** classical classifiers + deep reference mapping + foundation models.
- **Datasets:** lung atlas (already have) + PBMC/immune.
- **Compute:** **Mac M5 Pro first** — do everything that works on CPU (and Apple **MPS**
  where a framework supports it). A **cloud GPU (AWS) run is plotted out but deferred
  to the end** (see `docs/AWS_GPU.md`); we decide whether the cost is justified once
  the CPU/MPS results show where GPU timing would actually change conclusions.
- **Framing:** comprehensive, neutral (accuracy, runtime, memory, scalability, robustness).
- **Repo:** a **separate companion repo `actinn-jax-benchmark`** (keeps `actinn-jax` light).

## Methods under test

Each method is wrapped behind a common adapter (see Harness). "Env" = the isolated
environment it runs in; "Device" = CPU or Apple MPS.

### Tier 1 — Classical classifiers (CPU, lightweight)
| Method | Lang/env | Notes |
|---|---|---|
| **actinn-jax** | py/jax | this project (CPU) |
| **ACTINN (original)** | py/TF1 | baseline to quantify the rewrite's speedup |
| **CellTypist** | py | logistic regression, fast, pretrained models available |
| **SingleR** | R | correlation to reference profiles |
| **scmap-cell / scmap-cluster** | R | projection + rejection |
| **scPred** | R | PCA + SVM |
| **SVM (linear, w/ rejection)** | py/sklearn | repeatedly a top accuracy method; strong baseline |
| **kNN on PCA** | py/sklearn | trivial baseline / lower bound |

### Tier 2 — Deep / probabilistic reference mapping
| Method | Lang/env | Device | Notes |
|---|---|---|---|
| **scANVI** | py/scvi-tools | MPS/CPU | semi-supervised label transfer |
| **scArches (scANVI)** | py/scvi-tools | MPS/CPU | incremental "architecture surgery" |
| **Symphony** | R | CPU | compressed reference, very fast/low-mem |
| **Azimuth** | R/Seurat | CPU | Seurat anchor transfer |

### Tier 3 — Foundation models (pretrained, heavy)
| Method | Lang/env | Device | Notes |
|---|---|---|---|
| **scGPT** | py/torch | MPS/CPU | fine-tune or zero-shot annotation head |
| **scBERT** | py/torch | MPS/CPU | top accuracy in bbae392 |
| **scDeepSort** | py/torch | MPS/CPU | GNN; top accuracy in bbae392 |
| **TOSICA** | py/torch | MPS/CPU | transformer w/ pathway tokens |

> **Honest caveat (must be reported):** these methods are GPU-native; timing them on
> Apple CPU/MPS reflects a *no-CUDA laptop deployment*, not their best-case hardware.
> We report device per method and never compare a CUDA number against an MPS number.
> Optional follow-up: a CUDA re-run of Tiers 2–3 on a Linux box for a second table.

## Datasets

| Dataset | Role | Source | Notes |
|---|---|---|---|
| **HCLA (Sikkema)** → **krasnow** | lung, cross-atlas | local (have) | 585k ref / 66k query; already validated |
| **krasnow** (within) | lung, intra-dataset CV | local | clean accuracy ceiling |
| **PBMCbench** | immune, inter-protocol | [Zenodo/CELLxGENE] | multiple protocols → batch-transfer stress test |
| **Zheng68k** | immune, intra-dataset | public | closely related T-cell subtypes (hard) |
| **Experimentally-labeled immune (Liu, ZhengSort)** | immune, gold labels | bbae392 supp. | non-computational ground truth, if obtainable |

**Splits**
- **Intra-dataset:** stratified 5-fold CV → clean accuracy ceiling.
- **Inter-dataset / cross-protocol:** train on A, predict B → realistic reference
  mapping; measures batch robustness and label-vocabulary transfer.

**Preprocessing:** raw counts in; each method runs its *own* native normalization /
HVG selection (we do not force a single pipeline — that would bias toward methods
matching it). Where a common gene space is needed for fairness, provide both
"native" and "shared-gene" variants.

## Metrics

### Accuracy
- Overall accuracy, **macro-F1** (rare-type sensitive), per-class F1, confusion matrices.
- Cohen's κ.
- **Ontology-aware (lineage-credited) concordance** for cross-dataset runs where label
  vocabularies differ — reuse the Cell Ontology scoring already built in
  `benchmark/verify_real.py` (pred ≡ truth if same or ancestor/descendant in CL).
- Rejection-aware: for methods with an "unassigned" option (SVM-rej, scmap, CellTypist,
  scGPT), report accuracy-on-assigned and %-rejected separately.

### Runtime & resources
- **Fit time** and **predict time** reported *separately* (reference mapping = train
  once, map many — the amortized predict cost is the headline for repeated use).
- **Peak memory** (RSS) sampled via `psutil` in the subprocess runner.
- **Scalability curves:** time & memory vs #reference cells and #query cells
  (subsample sweep: 5k / 25k / 100k / 250k / full).
- Device (CPU/MPS) and thread cap recorded per run.

### Robustness (comprehensive framing)
- Cross-protocol accuracy drop vs intra-dataset (batch robustness).
- Macro-F1 vs overall accuracy gap (rare-type behavior).
- **Unseen cell type:** hold a type out of the reference; measure rejection vs
  forced-misassignment. Separates rejection-capable methods from the rest.

## Harness design

**Recommendation: a separate companion repo `actinn-jax-benchmark`**, not inside
`actinn-jax` — the benchmark pulls in R, scvi-tools, torch, and pretrained weights,
which would bloat a deliberately lightweight library. `actinn-jax` stays a clean dep.

**Adapter interface** (uniform across all methods):
```python
class AnnotationMethod:
    name: str; tier: str; env: str; device: str
    def fit(self, ref_h5ad: str, label_key: str) -> None: ...     # timed, mem-tracked
    def predict(self, query_h5ad: str) -> "Predictions": ...      # timed, mem-tracked
# Predictions: labels, optional probabilities, optional 'unassigned' flag
```

**Subprocess isolation.** Each method runs in its **own conda/venv env** (pinned),
invoked by the driver via subprocess with a standard CLI:
`run_method.py --method scanvi --ref ref.h5ad --query q.h5ad --label cell_type --out preds.parquet --metrics timing.json`.
This sidesteps dependency conflicts (R vs py, jax vs torch vs TF) entirely. I/O is
standardized: **h5ad in, parquet predictions + json metrics out.**

**Driver.** Config-driven (YAML) over the cartesian product
`{datasets} × {splits} × {methods} × {subsample sizes}`; writes one tidy results
row per run (method, dataset, split, n_ref, n_query, device, fit_s, predict_s,
peak_mem_mb, accuracy, macro_f1, kappa, onto_concordance, pct_rejected, seed).

**Timing fairness controls.**
- Discard a warm-up run (imports / JIT / weight load), then take the **median of N≥3**.
- Cap threads equally across methods (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, etc.).
- Separate the timed region from data loading; report fit and predict distinctly.
- Fixed seeds; pinned per-method environments captured in lockfiles.

## Reporting / deliverables
- Master tidy results table (parquet + CSV).
- **Accuracy × runtime Pareto** scatter per dataset (the headline figure).
- Scalability log-log plots (time & memory vs N).
- Memory bar charts; robustness panels (batch drop, rare-type, unseen-type).
- Leaderboard + written analysis; reproducible figure notebook.

## Phasing (so we stay in the loop between stages)
- **Phase 0 — skeleton:** adapter API + subprocess driver + results schema, wired for
  **2 methods (actinn-jax, CellTypist) on the lung dataset**, end-to-end. Validates the
  whole pipeline cheaply.
- **Phase 1 — Tier 1 + PBMC:** all classical methods (incl. R via isolated env) on lung
  + PBMC; intra- and inter-dataset splits. Mostly CPU, manageable.
- **Phase 2 — Tier 2:** scANVI, scArches, Symphony, Azimuth in isolated envs (MPS where
  supported).
- **Phase 3 — Tier 3:** scGPT, scBERT, scDeepSort, TOSICA on MPS with subsampled queries
  (+ pretrained-weight download / gene-vocab matching).
- **Phase 4 — scaling & robustness:** subsample sweeps, unseen-type and batch
  experiments, final figures + leaderboard.

## Key risks & mitigations
- **CPU/MPS-only timing** disadvantages GPU-native methods → document device per method;
  offer optional CUDA re-run as a separate table. Never cross-compare devices.
- **Dependency hell** (R/py/jax/torch/TF) → strict per-method subprocess isolation.
- **Label-vocabulary mismatch** cross-dataset → ontology-aware concordance (already built).
- **Foundation-model setup** (weights, gene vocab) → budget time in Phase 3; subsample.
- **Memory** (HCLA 585k on 48 GB) → reference subsampling sweeps; document caps.
- **Fairness of "native vs shared gene space"** → report both variants where it matters.

## Open data-acquisition tasks
- PBMCbench / Zheng68k / experimentally-labeled immune sets: locate canonical
  download (CELLxGENE, Zenodo, or bbae392 supplement) and convert to raw-count h5ad.
