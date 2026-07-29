"""Upload the reference archives to a Zenodo deposition.

    export ZENODO_TOKEN=...            # or write it to ~/.zenodo_token (chmod 600)
    python tools/zenodo_upload.py                    # create a DRAFT and upload
    python tools/zenodo_upload.py --publish          # ...and publish it (irreversible)
    python tools/zenodo_upload.py --deposition 1234  # add files to an existing draft

By default this stops at a draft: publishing mints a permanent DOI and a published Zenodo
record cannot be deleted, so the last step is left as a deliberate act. The draft URL is
printed for review.

Needs a token with the ``deposit:write`` and ``deposit:actions`` scopes, from
https://zenodo.org/account/settings/applications/tokens/new/

After publishing, set ``DEFAULT_BASE_URL`` in ``actinn_jax/registry.py`` to
``https://zenodo.org/records/<id>/files`` and commit.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from actinn_jax import registry  # noqa: E402

API = "https://zenodo.org/api"

DESCRIPTION = """
<p>Pre-trained cell-type annotation references for
<a href="https://github.com/iandriver/actinn-jax">actinn-jax</a>, a fast CPU-only
reference-mapping classifier for single-cell RNA-seq. Each archive unpacks to a
<code>HierarchicalReferenceModel</code> directory loadable with
<code>actinn_jax.bundled_reference(name)</code>, which downloads and caches them
automatically on first use.</p>

<ul>
<li><b>broad_human_v1</b> — census-wide human, ~800 cell types / 314 tissues; coarse
hierarchy from scPRINT embeddings.</li>
<li><b>broad_mouse_v1</b> — census-wide mouse, 453 cell types / 85 tissues; coarse
hierarchy from Cell Ontology lineage (no GPU in the build). On two datasets held out of
the reference entirely: 0.638 ontology-aware concordance, 0.718 over the 71% of cells
kept at min_prob=0.5.</li>
<li><b>panhuman_distill_v1</b> — human, 324 cell types, distilled from Pan-human
Azimuth; on a withheld liver study it scores 0.406 ontology concordance against 0.338
for broad_human_v1, at roughly three times its throughput.</li>
<li><b>liver_hlica_v1 / v2</b> — focused human liver references (38 and 48 types) from
the Human Liver Cell Atlas, for the refinement tier of the workflow.</li>
</ul>

<p>All references are built from CELLxGENE Census (release 2025-11-08). Build scripts and
the benchmark behind them:
<a href="https://github.com/iandriver/actinn-jax-benchmark">actinn-jax-benchmark</a>.</p>

<p><b>Attribution.</b> <i>panhuman_distill_v1</i> is derived from <b>Pan-human Azimuth</b>
(Sarkar, Li, Molla, ... Satija, bioRxiv 2026, doi:10.64898/2026.07.16.738997), whose model
weights are &copy; the authors under CC BY 4.0 (doi:10.5281/zenodo.20401417), obtained via
the MIT-licensed panhumanpy package. The liver references derive from the Human Liver Cell
Atlas (Edgar et al. 2026, CC BY 4.0, doi:10.64898/2026.06.30.735539).</p>
""".strip()

METADATA = {
    "metadata": {
        "title": "actinn-jax pre-trained cell-type annotation references "
                 "(human and mouse)",
        "upload_type": "dataset",
        "description": DESCRIPTION,
        "creators": [{"name": "Driver, Ian"}],
        "license": "cc-by-4.0",
        "keywords": ["single-cell RNA-seq", "cell type annotation", "reference mapping",
                     "CELLxGENE Census", "Cell Ontology", "actinn-jax"],
        "related_identifiers": [
            {"identifier": "https://github.com/iandriver/actinn-jax",
             "relation": "isSupplementTo", "scheme": "url"},
            {"identifier": "https://github.com/iandriver/actinn-jax-benchmark",
             "relation": "isSupplementTo", "scheme": "url"},
            {"identifier": "10.64898/2026.07.16.738997",
             "relation": "isDerivedFrom", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.20401417",
             "relation": "isDerivedFrom", "scheme": "doi"},
            {"identifier": "10.64898/2026.06.30.735539",
             "relation": "isDerivedFrom", "scheme": "doi"},
        ],
    }
}


def token():
    tok = os.environ.get("ZENODO_TOKEN")
    if tok:
        return tok.strip()
    path = os.path.expanduser("~/.zenodo_token")
    if os.path.exists(path):
        return open(path).read().strip()
    raise SystemExit(
        "No Zenodo token. Create one with the 'deposit:write' and 'deposit:actions' "
        "scopes at https://zenodo.org/account/settings/applications/tokens/new/ then\n"
        "  export ZENODO_TOKEN=...\n"
        "or write it to ~/.zenodo_token (chmod 600). The token is never printed or "
        "committed."
    )


def api(method, path, tok, data=None, raw=None, content_type="application/json"):
    url = f"{API}{path}{'&' if '?' in path else '?'}access_token={tok}"
    body = raw if raw is not None else (json.dumps(data).encode() if data else None)
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req) as r:
            text = r.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:600]
        raise SystemExit(f"Zenodo {method} {path} failed: {e.code}\n{detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/actinn_reference_release")
    ap.add_argument("--deposition", default=None, help="add to an existing draft id")
    ap.add_argument("--publish", action="store_true",
                    help="publish immediately (mints a permanent DOI; irreversible)")
    a = ap.parse_args()

    tok = token()
    archives = sorted(f for f in os.listdir(a.dir) if f.endswith(".tar.gz"))
    if not archives:
        raise SystemExit(f"no .tar.gz archives in {a.dir}; run package_references.py first")

    # Every archive must match the checksum the shipped package will verify against, or
    # users get a download the library then refuses to install.
    for f in archives:
        name = f[: -len(".tar.gz")]
        expected = registry.REGISTRY.get(name, {}).get("sha256")
        got = registry._sha256(os.path.join(a.dir, f))
        if expected != got:
            raise SystemExit(
                f"{f}: checksum {got} does not match the registry's {expected}. "
                "Re-run tools/package_references.py so the two agree before uploading."
            )
    print(f"{len(archives)} archives verified against the registry")

    if a.deposition:
        dep = api("GET", f"/deposit/depositions/{a.deposition}", tok)
    else:
        dep = api("POST", "/deposit/depositions", tok, data=METADATA)
        print(f"created draft deposition {dep['id']}")
    bucket = dep["links"]["bucket"]

    for f in archives:
        path = os.path.join(a.dir, f)
        size = os.path.getsize(path) / 1e6
        print(f"  uploading {f} ({size:.1f} MB) ...", flush=True)
        # File uploads go to the deposition's bucket URL, not the API root.
        with open(path, "rb") as fh:
            req = urllib.request.Request(f"{bucket}/{f}?access_token={tok}",
                                         data=fh.read(), method="PUT")
            req.add_header("Content-Type", "application/octet-stream")
            try:
                with urllib.request.urlopen(req) as r:
                    r.read()
            except urllib.error.HTTPError as e:
                raise SystemExit(f"upload of {f} failed: {e.code}\n"
                                 f"{e.read().decode()[:400]}")

    dep = api("GET", f"/deposit/depositions/{dep['id']}", tok)
    print(f"\ndraft: {dep['links'].get('html', dep['links']['self'])}")
    print(f"files: {[f['filename'] for f in dep.get('files', [])]}")

    if not a.publish:
        print("\nNot published. Review the draft, then either press Publish there or "
              f"re-run with --publish --deposition {dep['id']}.")
        return 0

    pub = api("POST", f"/deposit/depositions/{dep['id']}/actions/publish", tok)
    record = pub.get("record_id") or pub["id"]
    print(f"\npublished: https://zenodo.org/records/{record}")
    print(f"DOI: {pub.get('doi')}")
    print(f"\nNow set DEFAULT_BASE_URL in actinn_jax/registry.py to "
          f"https://zenodo.org/records/{record}/files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
