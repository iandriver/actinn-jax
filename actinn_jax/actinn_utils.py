"""JAX implementation of the ACTINN neural-network classifier.

This is a from-scratch reimplementation of the original TensorFlow 1.x ACTINN
model (Ma et al., Bioinformatics 2019). The network is a small 4-layer MLP
(LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX).

Everything here works in the standard machine-learning orientation:
``X`` has shape ``(n_samples, n_features)`` (i.e. cells x genes), which is the
transpose of the original ACTINN convention. Weights are applied as ``X @ W``.

The model is tiny, so JAX (JIT + autodiff) gives a fast, dependency-light
training loop that runs on CPU everywhere and on the Apple GPU via ``jax-metal``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.sparse as sp

# Network architecture: hidden layer widths, matching the original ACTINN.
LAYER_SIZES = (100, 50, 25)

# Default training hyper-parameters (mirror the original implementation).
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_NUM_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_L2 = 0.005
DEFAULT_SEED = 3


def initialize_parameters(num_features, num_types, layer_sizes=LAYER_SIZES, seed=DEFAULT_SEED):
    """Glorot-normal initialised weights / zero biases for the 4-layer MLP.

    Returns a dict of jnp arrays: ``W1..W4`` and ``b1..b4``. Weights are stored
    in ``(fan_in, fan_out)`` orientation so the forward pass is ``X @ W``.
    """
    sizes = [num_features, *layer_sizes, num_types]
    init = jax.nn.initializers.glorot_normal()
    key = jax.random.PRNGKey(seed)
    params = {}
    for i in range(len(sizes) - 1):
        key, subkey = jax.random.split(key)
        # glorot_normal expects (fan_out, fan_in); store transposed for X @ W.
        w = init(subkey, (sizes[i + 1], sizes[i]), jnp.float32).T
        params[f"W{i + 1}"] = w
        params[f"b{i + 1}"] = jnp.zeros((sizes[i + 1],), dtype=jnp.float32)
    return params


def forward(params, X):
    """Forward pass returning raw logits of shape ``(n_samples, n_types)``."""
    n_layers = len(params) // 2
    h = X
    for i in range(1, n_layers):
        h = jax.nn.relu(h @ params[f"W{i}"] + params[f"b{i}"])
    return h @ params[f"W{n_layers}"] + params[f"b{n_layers}"]


def _loss(params, X, Y, l2):
    logits = forward(params, X)
    ce = optax.softmax_cross_entropy(logits, Y).mean()
    # Mirror tf.nn.l2_loss == sum(w**2) / 2 over the weight matrices.
    reg = sum(jnp.sum(w ** 2) for k, w in params.items() if k.startswith("W"))
    return ce + l2 * 0.5 * reg


def _make_schedule(starting_learning_rate):
    # Exponential decay every 1000 optimizer steps, matching the original.
    return optax.exponential_decay(
        init_value=starting_learning_rate,
        transition_steps=1000,
        decay_rate=0.95,
        staircase=True,
    )


def train(
    X,
    Y,
    starting_learning_rate=DEFAULT_LEARNING_RATE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    l2=DEFAULT_L2,
    layer_sizes=LAYER_SIZES,
    seed=DEFAULT_SEED,
    print_cost=True,
    scale=None,
):
    """Train the MLP.

    Parameters
    ----------
    X : array or scipy.sparse matrix (n_samples, n_features)
        Scaled expression matrix (cells x genes). May be sparse: minibatches are
        densified one at a time, so peak memory scales with (batch_size x features)
        rather than (n_samples x features) -- important on atlas-scale references
        where the kept-gene matrix is nearly as wide as the full one.
    Y : array (n_samples, n_types)
        One-hot encoded labels.

    Returns
    -------
    dict of numpy arrays
        Trained parameters (W1..W4, b1..b4).
    """
    sparse = sp.issparse(X)
    if not sparse:
        X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    n_samples, n_features = X.shape
    n_types = Y.shape[1]

    # Optional per-gene standardization applied per minibatch (after row densification)
    # so peak memory stays at (batch_size x features): subtracting the mean would
    # otherwise destroy sparsity and force the whole matrix dense.
    _sc = None
    if scale is not None:
        _mu = np.asarray(scale[0], dtype=np.float32)
        _sd = np.asarray(scale[1], dtype=np.float32)
        _sd = np.where(_sd == 0.0, 1.0, _sd)
        _sc = (_mu, _sd)

    params = initialize_parameters(n_features, n_types, layer_sizes, seed)
    optimizer = optax.adam(_make_schedule(starting_learning_rate))
    opt_state = optimizer.init(params)

    n_batches = max(1, n_samples // batch_size)
    grad_fn = jax.value_and_grad(partial(_loss, l2=l2))

    @jax.jit
    def update(params, opt_state, xb, yb):
        loss, grads = grad_fn(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    def densify(rows):
        xb = X[rows]
        xb = np.asarray(xb.toarray() if sparse else xb, dtype=np.float32)
        if _sc is not None:
            xb = (xb - _sc[0]) / _sc[1]
        return xb

    key = jax.random.PRNGKey(seed)
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(subkey, n_samples))
        epoch_cost = 0.0
        for b in range(n_batches):
            rows = perm[b * batch_size:(b + 1) * batch_size]
            xb = jnp.asarray(densify(rows))
            yb = jnp.asarray(Y[rows])
            params, opt_state, loss = update(params, opt_state, xb, yb)
            epoch_cost += float(loss) / n_batches
        if print_cost and (epoch + 1) % 5 == 0:
            print("Cost after epoch %i: %f" % (epoch + 1, epoch_cost))

    if print_cost:
        # Training accuracy, computed in batches to avoid densifying all of X.
        correct = total = 0
        for b in range(n_batches + 1):
            rows = np.arange(b * batch_size, min((b + 1) * batch_size, n_samples))
            if len(rows) == 0:
                continue
            pred = np.asarray(predict_labels(params, densify(rows)))
            correct += int((pred == Y[rows].argmax(axis=1)).sum())
            total += len(rows)
        print("Parameters have been trained!")
        print("Train Accuracy:", correct / max(total, 1))
    return {k: np.asarray(v) for k, v in params.items()}


@jax.jit
def _probabilities(params, X):
    return jax.nn.softmax(forward(params, X), axis=1)


def predict_proba(params, X):
    """Return the full softmax probability matrix ``(n_samples, n_types)``."""
    params = {k: jnp.asarray(v) for k, v in params.items()}
    return np.asarray(_probabilities(params, jnp.asarray(X, dtype=jnp.float32)))


def predict_labels(params, X):
    """Return predicted class indices ``(n_samples,)`` (used internally)."""
    return jnp.argmax(forward(params, jnp.asarray(X, dtype=jnp.float32)), axis=1)


def predict(params, X):
    """Single forward pass returning ``(labels, max_probability)``.

    Replaces the two separate TF sessions of the original implementation: both
    the argmax label and its softmax confidence come from one pass.
    """
    proba = predict_proba(params, X)
    labels = np.argmax(proba, axis=1)
    max_prob = proba[np.arange(proba.shape[0]), labels]
    return labels, max_prob


def one_hot(labels, num_types):
    """One-hot encode integer labels as ``(n_samples, n_types)`` (no TF session)."""
    return np.asarray(jax.nn.one_hot(np.asarray(labels), num_types), dtype=np.float32)
