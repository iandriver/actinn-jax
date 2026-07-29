"""Command-line entry point: annotate an h5ad without writing any Python.

    actinn-jax references                      # what is bundled, and for which organism
    actinn-jax annotate query.h5ad             # picks the reference by organism
    actinn-jax annotate query.h5ad --reference broad_mouse_v1 --min-prob 0.5

With no ``--reference`` the organism is inferred from the query's Ensembl gene ids and the
matching bundled reference is used. Inference fails on symbol-keyed data, where the answer
is genuinely ambiguous (human ``CD3E`` vs mouse ``Cd3e`` differ only by case), so that case
asks for ``--reference`` rather than guessing.
"""

import argparse
import sys


def _cmd_references(args):
    from .hierarchy import list_references

    refs = list_references()
    if not refs:
        print("No references found.")
        return 1
    w = max(len(r["name"]) for r in refs)
    print(f"{'name':<{w}}  {'organism':<14} {'types':>6} {'MB':>6}  {'status':<10} source")
    print("-" * (w + 62))
    for r in refs:
        print(f"{r['name']:<{w}}  {r['organism']:<14} {r['n_classes']:>6} "
              f"{r['size_mb']:>6}  {r['status']:<10} {r['source']}")
    print("\n'available' downloads on first use; 'cached' is already downloaded.")
    return 0


def _cmd_fetch(args):
    """Pre-download references, so a tutorial or an offline session does not stall on a
    download partway through."""
    from . import registry

    names = args.names or sorted(registry.REGISTRY)
    for name in names:
        path = registry.fetch_reference(name, force=args.force)
        print(f"{name}: {path}")
    return 0


def _cmd_cache(args):
    from . import registry

    if args.clear:
        removed = registry.clear_cache(args.name)
        print(f"removed: {', '.join(removed) if removed else 'nothing'}")
        return 0
    print(f"cache: {registry.cache_dir()}")
    cached = registry.cached_references()
    print(f"cached references: {', '.join(cached) if cached else 'none'}")
    return 0


def _cmd_annotate(args):
    import scanpy as sc

    from .hierarchy import (annotate, bundled_reference, default_reference_for,
                            infer_organism, list_references)

    adata = sc.read_h5ad(args.query)
    print(f"query: {adata.shape}", file=sys.stderr)

    name = args.reference
    if name is None:
        organism = infer_organism(adata.var_names)
        if organism is None and getattr(adata, "raw", None) is not None:
            organism = infer_organism(adata.raw.var_names)
        if organism is None:
            avail = ", ".join(r["name"] for r in list_references())
            raise SystemExit(
                "Could not tell which organism this query is from (its gene ids are not "
                f"Ensembl). Pass --reference explicitly. Available: {avail}"
            )
        name = default_reference_for(organism)
        if name is None:
            raise SystemExit(f"No bundled reference for {organism}.")
        print(f"organism {organism} -> reference {name}", file=sys.stderr)

    model = bundled_reference(name)
    adata = annotate(adata, model, min_prob=args.min_prob)

    cols = ["celltype", "celltype_probability", "celltype_coarse"]
    out = adata.obs[[c for c in cols if c in adata.obs]]
    out.to_csv(args.output)
    print(f"wrote {args.output} ({len(out)} cells)", file=sys.stderr)
    if args.write_h5ad:
        adata.write_h5ad(args.write_h5ad)
        print(f"wrote {args.write_h5ad}", file=sys.stderr)

    counts = adata.obs["celltype"].value_counts()
    print(f"\ntop labels ({len(counts)} distinct):", file=sys.stderr)
    for label, n in counts.head(10).items():
        print(f"  {n:>7}  {label}", file=sys.stderr)
    if args.min_prob:
        unknown = int((adata.obs["celltype"] == "unknown").sum())
        print(f"\nabstained at min_prob={args.min_prob}: {unknown} cells "
              f"({unknown / adata.n_obs:.1%})", file=sys.stderr)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="actinn-jax", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("references", help="list references and their status").set_defaults(
        func=_cmd_references)

    f = sub.add_parser("fetch", help="pre-download references into the cache")
    f.add_argument("names", nargs="*", help="reference names (default: all downloadable)")
    f.add_argument("--force", action="store_true", help="re-download even if cached")
    f.set_defaults(func=_cmd_fetch)

    c = sub.add_parser("cache", help="show or clear the download cache")
    c.add_argument("--clear", action="store_true")
    c.add_argument("--name", default=None, help="clear just this one")
    c.set_defaults(func=_cmd_cache)

    a = sub.add_parser("annotate", help="annotate an h5ad against a bundled reference")
    a.add_argument("query", help="path to an .h5ad of raw counts")
    a.add_argument("--reference", "-r", default=None,
                   help="bundled reference name (default: chosen by organism)")
    a.add_argument("--min-prob", type=float, default=None,
                   help="abstain threshold; cells below it are labeled 'unknown'")
    a.add_argument("--output", "-o", default="annotations.csv")
    a.add_argument("--write-h5ad", default=None,
                   help="also write the annotated AnnData here")
    a.set_defaults(func=_cmd_annotate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
