"""Download-on-first-use for the large bundled references.

The census-scale references are ~35-50 MB each and there are several, which puts a wheel
carrying all of them at 110 MB -- over PyPI's 100 MB per-file limit, and growing with every
reference added. Only the smallest, best general-purpose human reference ships inside the
package; the rest are fetched once from an archive and cached, so

    actinn_jax.bundled_reference("broad_mouse_v1")

works whether or not it has been downloaded before. Nothing here needs the user to think
about where models live.

Archive location is configurable (``ACTINN_JAX_REFERENCE_URL``) so the same code works
against a Zenodo record, a GitHub release, or an internal mirror. Each archive is verified
against a recorded SHA-256 before it is unpacked: a truncated download that silently
produced a half-model would be worse than a failed one.

Cache: ``$ACTINN_JAX_CACHE`` / ``$XDG_CACHE_HOME/actinn-jax`` / ``~/.cache/actinn-jax``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

# Base URL for reference archives. Overridable so a mirror or a pre-publication staging
# location can be used without touching the code.
DEFAULT_BASE_URL = os.environ.get(
    "ACTINN_JAX_REFERENCE_URL",
    "https://zenodo.org/records/21688151/files",
)

# Cite the concept DOI (10.5281/zenodo.21688150) to always resolve to the latest version;
# 10.5281/zenodo.21688151 pins this exact set of references.
REFERENCE_DOI = "10.5281/zenodo.21688151"

# name -> metadata. `sha256` is filled in by tools/package_references.py when the archives
# are built; an entry whose checksum is None cannot be fetched (we refuse to install an
# unverifiable model rather than trust the network).
REGISTRY = {
    "broad_human_v1": {
        "organism": "homo_sapiens",
        "n_classes": 798,
        "size_mb": 45,
        "sha256": "541bbbcdc36826b10ee2f141511e4e9fd5cb0a5529f21e799fad06a9d7e465a4",
        "description": "Census-wide human reference, ~800 types, scPRINT-derived hierarchy",
    },
    "broad_mouse_v1": {
        "organism": "mus_musculus",
        "n_classes": 453,
        "size_mb": 34,
        "sha256": "b013364b214185597c6b20449de93ea9e5af4b27e0ea1718cd70a4ea42abfdf0",
        "description": "Census-wide mouse reference, 453 types, Cell-Ontology hierarchy",
    },
    "liver_hlica_v1": {
        "organism": "homo_sapiens",
        "n_classes": 38,
        "size_mb": 11,
        "sha256": "46a9c93286bc6c4e8538aa50cc48a53b74236479d7d8859c57a695cc705c97d2",
        "description": "HLiCA liver reference (38 types)",
    },
    "liver_hlica_v2": {
        "organism": "homo_sapiens",
        "n_classes": 48,
        "size_mb": 11,
        "sha256": "a94ba60c6402bb1d489bc98bf418f5e1bb0ccbaf5e20273a71c8792ef6ad8fa5",
        "description": "HLiCA liver reference (48 types)",
    },
    # Also ships inside the wheel, so this entry is normally unused -- it exists so the
    # archive record is complete and so a user who installs from a trimmed build (or
    # deletes the bundled copy) can still fetch it.
    "panhuman_distill_v1": {
        "organism": "homo_sapiens",
        "n_classes": 324,
        "size_mb": 15,
        "sha256": "0967833760761d9362c6a5cf9ffbcf97132ddd05f6780d73917dbebc3adec608",
        "description": "Human reference distilled from Pan-human Azimuth, 324 types",
    },
}

# Shipped inside the wheel: small, and the most accurate general-purpose human reference,
# so `pip install actinn-jax` annotates human data offline with no download.
BUNDLED = ("panhuman_distill_v1",)


def cache_dir():
    root = os.environ.get("ACTINN_JAX_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "actinn-jax")
    return os.path.join(root, "references")


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url, dest, quiet=False):
    # urlretrieve calls the hook once per block -- thousands of times for a 35 MB file --
    # so only emit when the whole percent actually changes, or a notebook fills with
    # progress lines.
    state = {"pct": -1}

    def hook(count, block_size, total):
        if quiet or total <= 0:
            return
        pct = min(100, count * block_size * 100 // total)
        if pct != state["pct"]:
            state["pct"] = pct
            print(f"\r  {pct:3d}%  {url.rsplit('/', 1)[-1]}", end="", file=sys.stderr,
                  flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    if not quiet:
        print(file=sys.stderr)


def fetch_reference(name, force=False, quiet=False, base_url=None):
    """Download and unpack a reference into the cache; return its directory.

    Returns immediately if it is already cached. Raises rather than installing an archive
    whose checksum does not match, or one for which no checksum is recorded.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"'{name}' is not a downloadable reference. Known: "
            f"{', '.join(sorted(REGISTRY))}"
        )
    target = os.path.join(cache_dir(), name)
    if os.path.isdir(target) and not force:
        return target

    meta = REGISTRY[name]
    expected = meta.get("sha256")
    if not expected:
        raise RuntimeError(
            f"No checksum is recorded for '{name}', so it cannot be verified after "
            "download. This build of actinn-jax predates publication of that archive — "
            "upgrade the package, or point ACTINN_JAX_REFERENCE_URL at a trusted mirror "
            "and supply the checksum."
        )

    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/{name}.tar.gz"
    os.makedirs(cache_dir(), exist_ok=True)
    if not quiet:
        print(f"actinn-jax: fetching {name} (~{meta['size_mb']} MB) from {url}",
              file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, f"{name}.tar.gz")
        _download(url, archive, quiet=quiet)
        got = _sha256(archive)
        if got != expected:
            raise RuntimeError(
                f"Checksum mismatch for {name}: expected {expected}, got {got}. "
                "The download is corrupt or the archive has changed; refusing to install it."
            )
        staged = os.path.join(tmp, "unpacked")
        with tarfile.open(archive) as tf:
            # Refuse absolute paths and parent traversal from an archive we are unpacking.
            for member in tf.getmembers():
                if member.name.startswith(("/", "..")) or ".." in member.name.split("/"):
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
            tf.extractall(staged)
        root = os.path.join(staged, name)
        if not os.path.isdir(root):          # archive without a top-level directory
            root = staged
        # Move into place only once it is complete, so an interrupted fetch cannot leave a
        # half-model that later looks cached.
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.move(root, target)

    return target


def cached_references():
    """Names already present in the local cache."""
    root = cache_dir()
    if not os.path.isdir(root):
        return []
    return sorted(n for n in os.listdir(root)
                  if os.path.exists(os.path.join(root, n, "manifest.json")))


def clear_cache(name=None):
    """Delete one cached reference, or all of them. Returns what was removed."""
    root = cache_dir()
    targets = [name] if name else cached_references()
    removed = []
    for n in targets:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            shutil.rmtree(d)
            removed.append(n)
    return removed
