"""Package the large references into archives for upload, and record their checksums.

Only the smallest reference ships inside the wheel; the rest are downloaded on first use
(``actinn_jax/registry.py``). This builds the archives to upload, computes their SHA-256,
and rewrites the checksums in the registry so the shipped package can verify what it
downloads.

    python tools/package_references.py                  # build archives + update registry
    python tools/package_references.py --check          # report, change nothing

Upload the resulting archives to the record named by ``ACTINN_JAX_REFERENCE_URL``
(a Zenodo record, a GitHub release, or a mirror), then commit the updated registry.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from actinn_jax import registry  # noqa: E402

REFS = os.path.join(ROOT, "actinn_jax", "references")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/actinn_reference_release")
    ap.add_argument("--check", action="store_true", help="report only; do not write")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for name in sorted(registry.REGISTRY):
        src = os.path.join(REFS, name)
        if not os.path.isdir(src):
            print(f"SKIP {name}: not present at {src}")
            continue
        archive = os.path.join(a.out, f"{name}.tar.gz")
        if not a.check:
            with tarfile.open(archive, "w:gz") as tf:
                tf.add(src, arcname=name)
        if not os.path.exists(archive):
            print(f"SKIP {name}: no archive (use without --check to build)")
            continue
        digest = sha256(archive)
        size = os.path.getsize(archive) / 1e6
        rows.append((name, digest, size))
        print(f"{name:<22} {size:>7.1f} MB  {digest}")

    if a.check or not rows:
        return

    # Rewrite the recorded checksums in place, leaving the rest of the registry untouched.
    path = os.path.join(ROOT, "actinn_jax", "registry.py")
    text = open(path).read()
    for name, digest, size in rows:
        block = re.search(rf'"{name}": \{{.*?\}},\n', text, re.S)
        if not block:
            print(f"WARN: no registry entry for {name}")
            continue
        updated = re.sub(r'"sha256": [^,]+,', f'"sha256": "{digest}",', block.group(0))
        updated = re.sub(r'"size_mb": [^,]+,', f'"size_mb": {round(size)},', updated)
        text = text.replace(block.group(0), updated)
    open(path, "w").write(text)
    print(f"\nupdated checksums in {path}")

    manifest = os.path.join(a.out, "manifest.json")
    with open(manifest, "w") as fh:
        json.dump([{"name": n, "sha256": d, "size_mb": round(s, 1)} for n, d, s in rows],
                  fh, indent=2)
    print(f"wrote {manifest}")
    print(f"\nUpload every *.tar.gz in {a.out} to the archive, then set "
          "ACTINN_JAX_REFERENCE_URL / DEFAULT_BASE_URL to the record's files URL.")


if __name__ == "__main__":
    sys.exit(main())
