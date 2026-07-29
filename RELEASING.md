# Releasing actinn-jax

Publishing runs on GitHub Actions via **PyPI Trusted Publishing** (OIDC). There is no API
token in GitHub secrets, on a laptop, or in `~/.pypirc`.

## One-time setup

At <https://pypi.org/manage/account/publishing/> (the *pending publisher* form, since this
works before the project exists on the index):

| Field | Value |
|---|---|
| PyPI Project Name | `actinn-jax` |
| Owner | `iandriver` |
| Repository name | `actinn-jax` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Optionally repeat at <https://test.pypi.org/manage/account/publishing/> with environment
`testpypi` to enable rehearsal runs.

Then create the two GitHub environments (Settings → Environments): `pypi` and `testpypi`.
Adding a required reviewer to `pypi` makes every release pause for an explicit approval,
which is worth it — a PyPI filename can never be reused, and yanking hides a release rather
than deleting it.

## Cutting a release

1. Bump `__version__` in `actinn_jax/__init__.py`. That is the **only** place a version
   lives; `pyproject.toml` reads it via `[tool.setuptools.dynamic]`.
2. Commit and push to `main`.
3. Rehearse if you want: Actions → *publish* → Run workflow → `testpypi`, then
   `pip install -i https://test.pypi.org/simple/ actinn-jax` in a scratch venv.
4. Tag and push:

   ```bash
   git tag v0.3.0 && git push --tags
   ```

The workflow tests on 3.10/3.11/3.12, builds both artifacts, runs `twine check`, enforces
the size limits, installs the wheel in a clean venv and confirms the bundled reference
loads offline — then publishes. Nothing publishes on a plain merge; only a `v*` tag does.

## What the size gate is protecting

The wheel and the sdist each grew to 110 MB once, for *different* reasons, and PyPI rejects
anything over 100 MB:

- the **wheel** picks up everything in the package directory unless
  `include-package-data = false` (pyproject.toml);
- the **sdist** ignores that setting entirely and needs `MANIFEST.in` to prune
  `actinn_jax/references`.

Only `panhuman_distill_v1` (~15 MB) is distributed. The census-scale references live on
Zenodo ([10.5281/zenodo.21688151](https://doi.org/10.5281/zenodo.21688151)) and are fetched
on first use. If CI fails on artifact size, one of those two lines has regressed.

## Publishing a new or rebuilt reference

The package verifies every download against a checksum, so an archive and a release have to
move together:

1. Rebuild the reference (see the benchmark repo's `docs/UPDATE_BROAD_REFERENCE.md`).
2. `python tools/package_references.py` — rebuilds the archives and rewrites the checksums
   in `actinn_jax/registry.py`.
3. `python tools/zenodo_upload.py` — uploads to a draft; publish a **new version** of the
   Zenodo record and update `DEFAULT_BASE_URL` to it.
4. Release the package.

Skipping step 3 leaves users downloading an archive whose checksum no longer matches, and
the library will refuse it — a loud failure, but a confusing one.
