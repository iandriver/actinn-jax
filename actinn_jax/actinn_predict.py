"""Fast, sparse-aware ACTINN reference mapping for single-cell data.

Two usage patterns are supported:

* **One-off run** -- :func:`celltype_predict_actinn` trains on a reference and
  predicts a query in a single call (drop-in for the original API, plus it writes
  the same ``.txt`` outputs).
* **Cached reference** -- :func:`train_reference` trains once and returns a
  :class:`ReferenceModel` you can ``.save()`` / ``.load()`` and reuse to
  :func:`predict` many query datasets without retraining.

Preprocessing never densifies the full matrix: gene-name matching and intersection
happen on the sparse matrix, and only the (much smaller) shared-gene submatrix is
materialised as dense for normalization.
"""

import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

import actinn_jax.actinn_utils as au

sc.settings.verbosity = 0

MIN_SHARED_GENES = 500


# --------------------------------------------------------------------------- #
# Sparse-aware preprocessing helpers
# --------------------------------------------------------------------------- #
def _as_csr(X):
    """Return ``X`` as a float32 scipy CSR matrix without a dense round-trip."""
    if sp.issparse(X):
        return X.tocsr().astype(np.float32, copy=False)
    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def _resolve_matrix(adata, use_raw):
    """Select the count matrix + gene names, preferring raw counts when present.

    ACTINN expects raw counts. CELLxGENE-style objects keep log-normalized values
    in ``.X`` and the raw integer counts in ``.raw``. ``use_raw='auto'`` (default)
    uses ``adata.raw`` when available, otherwise ``adata.X``.
    """
    if use_raw == "auto":
        use_raw = adata.raw is not None
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        return adata.raw.X, pd.Index(adata.raw.var_names)
    return adata.X, pd.Index(adata.var_names)


def _extract(adata, label_name=None, use_raw="auto"):
    """Pull counts + de-duplicated upper-cased gene names (and labels) from AnnData.

    Gene names are upper-cased and de-duplicated (first occurrence kept) using
    vectorized pandas ops -- no Python loops. Returns ``(csr, gene_index, labels)``.
    """
    X, genes = _resolve_matrix(adata, use_raw)
    genes = genes.str.upper()
    keep = ~genes.duplicated(keep="first")
    X = _as_csr(X)[:, np.asarray(keep)]
    genes = genes[keep]
    labels = None if label_name is None else np.asarray(adata.obs[label_name].values)
    return X, genes, labels


def _normalize(dense):
    """Library-size normalize each cell to 1e4, then ``log2(x + 1)`` (in place)."""
    lib = dense.sum(axis=1, keepdims=True)
    np.divide(dense, lib, out=dense, where=lib > 0)
    dense *= 1e4
    return np.log2(dense + 1, out=dense)


def _gene_filter(normalized):
    """Boolean mask of genes kept by ACTINN's expr- and CV-percentile filters.

    ``normalized`` is dense ``(n_cells, n_genes)``. Genes are kept when both their
    summed expression and their coefficient of variation fall within the 1st-99th
    percentile range (computed across all supplied cells), matching the original.
    """
    expr = np.nansum(normalized, axis=0)
    keep = (expr >= np.percentile(expr, 1)) & (expr <= np.percentile(expr, 99))

    sub = normalized[:, keep]
    mean = sub.mean(axis=0)
    std = sub.std(axis=0)
    cv = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
    cv_keep = (cv >= np.percentile(cv, 1)) & (cv <= np.percentile(cv, 99))

    mask = np.zeros(normalized.shape[1], dtype=bool)
    mask[np.where(keep)[0][cv_keep]] = True
    return mask


def _project(X, gene_index, target_genes):
    """Dense ``(n_cells, len(target_genes))`` with columns aligned to ``target_genes``.

    Genes present in ``gene_index`` are copied; genes missing from the query are
    left as zeros. This is how a query is projected onto a reference's fixed gene
    set. ``gene_index`` must be unique (it is, after de-duplication).
    """
    pos = gene_index.get_indexer(pd.Index(target_genes))
    out = np.zeros((X.shape[0], len(target_genes)), dtype=np.float32)
    present = pos >= 0
    if present.any():
        out[:, present] = X[:, pos[present]].toarray()
    return out


def _encode_labels(labels):
    """Return ``(int_labels, classes)`` with deterministic, sorted class order."""
    classes, inverse = np.unique(labels, return_inverse=True)
    return inverse, [str(c) for c in classes]


# --------------------------------------------------------------------------- #
# Cached reference model
# --------------------------------------------------------------------------- #
class ReferenceModel:
    """A trained ACTINN model plus everything needed to map a new query.

    Attributes
    ----------
    params : dict of numpy arrays
        Trained network weights / biases.
    norm_genes : list of str
        Gene set over which library-size normalization is computed.
    select_idx : numpy array
        Positions (within ``norm_genes``) of the genes fed to the network.
    classes : list of str
        Cell-type names, indexed by the network's output units.
    """

    def __init__(self, params, norm_genes, select_idx, classes):
        self.params = params
        self.norm_genes = list(norm_genes)
        self.select_idx = np.asarray(select_idx)
        self.classes = list(classes)

    def _features_block(self, X, genes):
        """Project + normalize a sparse count block onto the reference gene space."""
        dense = _project(X, genes, self.norm_genes)
        _normalize(dense)
        return dense[:, self.select_idx]

    def predict_proba(self, adata, use_raw="auto", chunk_size=50000):
        """Softmax probabilities ``(n_cells, n_types)``, computed in cell chunks.

        Chunking bounds peak memory so atlas-scale queries (hundreds of thousands
        of cells) never materialise as one giant dense matrix.
        """
        X, genes, _ = _extract(adata, use_raw=use_raw)
        n = X.shape[0]
        out = np.empty((n, len(self.classes)), dtype=np.float32)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            feats = self._features_block(X[start:stop], genes)
            out[start:stop] = au.predict_proba(self.params, feats)
        return out

    def predict_frame(self, adata, use_raw="auto", chunk_size=50000):
        """Return a tidy DataFrame of predicted label + probability, indexed by cell."""
        proba = self.predict_proba(adata, use_raw=use_raw, chunk_size=chunk_size)
        idx = np.argmax(proba, axis=1)
        frame = pd.DataFrame(
            {
                "celltype": [self.classes[i] for i in idx],
                "celltype_probability": proba[np.arange(len(idx)), idx],
            },
            index=list(adata.obs_names),
        )
        return frame, proba

    # -- persistence -------------------------------------------------------- #
    @staticmethod
    def _paths(path):
        prefix = path[:-4] if path.endswith((".npz", ".json")) else path
        return prefix + ".npz", prefix + ".json"

    def save(self, path):
        """Save weights to ``{path}.npz`` and metadata to ``{path}.json``."""
        npz_path, json_path = self._paths(path)
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        np.savez(npz_path, **self.params, select_idx=self.select_idx)
        with open(json_path, "w") as fh:
            json.dump({"norm_genes": self.norm_genes, "classes": self.classes}, fh)
        return npz_path, json_path

    @classmethod
    def load(cls, path):
        """Load a model previously written by :meth:`save`."""
        npz_path, json_path = cls._paths(path)
        data = np.load(npz_path)
        params = {k: data[k] for k in data.files if k.startswith(("W", "b"))}
        with open(json_path) as fh:
            meta = json.load(fh)
        return cls(params, meta["norm_genes"], data["select_idx"], meta["classes"])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _load_adata(data):
    return sc.read_h5ad(data) if isinstance(data, str) else data


def _subsample_indices(labels, max_per_label, seed=0):
    """Balanced per-label subsampling: at most ``max_per_label`` cells per class."""
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) > max_per_label:
            idx = rng.choice(idx, max_per_label, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def train_reference(
    train_data,
    train_label_name="celltype",
    learning_rate=au.DEFAULT_LEARNING_RATE,
    num_epochs=au.DEFAULT_NUM_EPOCHS,
    batch_size=au.DEFAULT_BATCH_SIZE,
    use_raw="auto",
    max_cells_per_label=None,
    seed=au.DEFAULT_SEED,
    print_cost=True,
):
    """Train an ACTINN model on a labeled reference and return a ReferenceModel.

    Parameters
    ----------
    train_data : str or AnnData
        Path to an ``.h5ad`` file (or an AnnData) with raw counts (in ``.raw`` or
        ``.X``) and cell-type labels in ``obs[train_label_name]``.
    use_raw : {'auto', True, False}
        Use ``adata.raw`` counts when available (default 'auto').
    max_cells_per_label : int, optional
        Balance and cap training set to this many cells per class. Strongly
        recommended for large atlases -- bounds memory and speeds up training.
    """
    adata = _load_adata(train_data)
    X, genes, labels = _extract(adata, train_label_name, use_raw=use_raw)

    if max_cells_per_label is not None:
        sel = _subsample_indices(labels, max_cells_per_label, seed)
        X, labels = X[sel], labels[sel]

    dense = X.toarray()
    _normalize(dense)
    mask = _gene_filter(dense)
    if mask.sum() < MIN_SHARED_GENES:
        raise ValueError(
            "Not enough informative genes after filtering "
            f"({int(mask.sum())} < {MIN_SHARED_GENES})."
        )

    select_idx = np.where(mask)[0]
    int_labels, classes = _encode_labels(labels)
    Y = au.one_hot(int_labels, len(classes))

    print("Cell types in training set:", {c: i for i, c in enumerate(classes)})
    print("# Training cells:", len(labels))
    params = au.train(
        dense[:, select_idx], Y,
        starting_learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        print_cost=print_cost,
    )
    return ReferenceModel(params, list(genes), select_idx, classes)


def predict(
    adata,
    reference_model,
    output_label_name="celltype",
    outpath=None,
    output_h5ad=False,
    use_raw="auto",
    chunk_size=50000,
):
    """Annotate ``adata`` using a trained :class:`ReferenceModel` (no retraining).

    Adds ``output_label_name`` and ``output_label_name + '_probability'`` to
    ``adata.obs`` (index-aligned), and optionally writes outputs to ``outpath``.
    Prediction runs in cell chunks of ``chunk_size`` to bound memory.
    """
    frame, proba = reference_model.predict_frame(
        adata, use_raw=use_raw, chunk_size=chunk_size
    )
    adata.obs[output_label_name] = frame.loc[adata.obs.index, "celltype"]
    adata.obs[output_label_name + "_probability"] = frame.loc[
        adata.obs.index, "celltype_probability"
    ]

    if outpath is not None:
        prob_df = pd.DataFrame(
            proba.T, index=reference_model.classes, columns=list(adata.obs_names)
        )
        prob_df.to_csv(
            os.path.join(outpath, "predicted_probabilities.txt"), sep="\t"
        )
        frame[["celltype"]].rename(columns={"celltype": output_label_name}).to_csv(
            os.path.join(outpath, output_label_name + "_predicted_label.txt"),
            sep="\t",
            index=False,
        )
        if output_h5ad:
            adata.write_h5ad(os.path.join(outpath, "predicted_label.h5ad"))
    return adata, reference_model


def celltype_predict_actinn(
    adata: AnnData,
    train_data_path: str,
    outpath: str,
    train_label_name: str = "celltype",
    output_label_name: str = "celltype",
    output_h5ad: bool = False,
    use_raw="auto",
):
    """One-off train-and-predict, mirroring the original public API.

    Genes are filtered jointly on the combined reference + query matrix (as in the
    original implementation) for maximum fidelity, then the model trains on the
    reference and predicts the query. Writes ``..._predicted_probabilities.txt``
    and ``..._predicted_label.txt`` to ``outpath``.

    Returns ``(adata, parameters)``.
    """
    Xq, genes_q, _ = _extract(adata, use_raw=use_raw)
    train_adata = _load_adata(train_data_path)
    Xr, genes_r, labels = _extract(train_adata, train_label_name, use_raw=use_raw)

    # Shared genes, intersected while still sparse.
    common = genes_r.intersection(genes_q)
    common = pd.Index(sorted(common))
    if len(common) < MIN_SHARED_GENES:
        raise ValueError(
            "Not enough shared genes: verify that gene names are the same format "
            f"({len(common)} < {MIN_SHARED_GENES})."
        )
    ref = _project(Xr, genes_r, common)
    qry = _project(Xq, genes_q, common)

    # Joint normalization + gene filtering across reference and query.
    combined = np.vstack([ref, qry])
    _normalize(combined)
    mask = _gene_filter(combined)
    combined = combined[:, mask]
    ref, qry = combined[: ref.shape[0]], combined[ref.shape[0]:]

    int_labels, classes = _encode_labels(labels)
    Y = au.one_hot(int_labels, len(classes))
    print("Cell types in training set:", {c: i for i, c in enumerate(classes)})
    print("# Training cells:", len(labels))
    params = au.train(ref, Y, num_epochs=au.DEFAULT_NUM_EPOCHS)

    proba = au.predict_proba(params, qry)
    idx = np.argmax(proba, axis=1)
    barcodes = list(adata.obs_names)

    train_save_name = os.path.basename(train_data_path).split(".h5ad")[0] \
        if isinstance(train_data_path, str) else "reference"
    prob_df = pd.DataFrame(proba.T, index=classes, columns=barcodes)
    prob_df.to_csv(
        os.path.join(outpath, train_save_name + "_predicted_probabilities.txt"),
        sep="\t",
    )

    pred = pd.DataFrame(
        {
            output_label_name: [classes[i] for i in idx],
            output_label_name + "_probability": proba[np.arange(len(idx)), idx],
        },
        index=barcodes,
    )
    pred[[output_label_name]].to_csv(
        os.path.join(outpath, output_label_name + "_predicted_label.txt"),
        sep="\t",
        index=False,
    )
    adata.obs[output_label_name] = pred.loc[adata.obs.index, output_label_name]
    adata.obs[output_label_name + "_probability"] = pred.loc[
        adata.obs.index, output_label_name + "_probability"
    ]
    if output_h5ad:
        adata.write_h5ad(os.path.join(outpath, "predicted_label.h5ad"))

    return adata, params
