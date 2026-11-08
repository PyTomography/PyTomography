"""Download and check the tutorial data from a shell.

    python -m pytomography.datasets list
    python -m pytomography.datasets info SPECT/Lu177-NEMA-SymT2
    python -m pytomography.datasets fetch SPECT/Lu177-PSMA-GEDisc CT/SophiaBeads-256
    python -m pytomography.datasets fetch --tutorial t_dicomdata
    python -m pytomography.datasets verify --hash
    python -m pytomography.datasets cache-key SPECT/Lu177-PSMA-GEDisc
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import (DatasetNotAvailable, DownloadError, _tutorial_datasets, available, cache_key, fetch, info, registry,
               verify)
from ._download import size_text


def _table(rows: List[List[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def main(argv: Optional[List[str]] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", help="the data folder (default: PYTOMOGRAPHY_DATA, or ~/pytomography_data)")
    parser = argparse.ArgumentParser(prog="python -m pytomography.datasets",
                                     description="Download and check the PyTomography tutorial data.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", parents=[common], help="every dataset, its size and licence, and its state")
    p = commands.add_parser("info", parents=[common], help="what a dataset is, its source, licence and parts")
    p.add_argument("name")
    p = commands.add_parser("fetch", parents=[common], help="download datasets, if they are not there yet")
    p.add_argument("names", nargs="*", metavar="name")
    p.add_argument("--tutorial", action="append", default=[], help="every dataset this tutorial notebook reads")
    p.add_argument("--all", action="store_true", help="every dataset that can be downloaded")
    p.add_argument("--extras", default="", help="comma-separated extras to add, or 'all'")
    p.add_argument("--verify", choices=["size", "hash", "none"], default="size",
                   help="how to check datasets that are already there (default: size)")
    p.add_argument("--force", action="store_true", help="download and unpack again")
    p.add_argument("--workers", type=int, default=4, help="most connections at once (default: 4)")
    p.add_argument("--quiet", action="store_true", help="print nothing but errors")
    p = commands.add_parser("verify", parents=[common], help="check downloaded datasets for missing or changed files")
    p.add_argument("names", nargs="*", metavar="name")
    p.add_argument("--hash", action="store_true", help="re-read every file and compare checksums, not only sizes")
    p = commands.add_parser("cache-key", parents=[common], help="a key that changes when these datasets change")
    p.add_argument("names", nargs="+", metavar="name")
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            columns = ["dataset", "download", "licence", "state"]
            print(_table([columns] + available(args.data_dir)[columns].values.tolist()))
        elif args.command == "info":
            d = info(args.name, args.data_dir)
            print(f"{d['name']}: {d['title']}")
            print(f"  source:    {d['source']} ({d['url']})")
            print(f"  licence:   {d['licence']}")
            if d["cite"]:
                print(f"  cite:      {d['cite']}")
            if d["status"] == "pending":
                print(f"  not downloadable yet: {d['note']}")
            print(f"  download:  {size_text(d['download_bytes'])}, {size_text(d['disk_bytes'])} on disk")
            for part in d["parts"]:
                print(f"  part:      {part}")
            for name, extra in d["extras"].items():
                print(f"  extra:     {name}: {extra['title']} ({size_text(extra['download_bytes'])})")
            print(f"  tutorials: {', '.join(d['tutorials'] or []) or 'none'}")
            print(f"  folder:    {d['folder']} ({d['state']})")
        elif args.command == "fetch":
            names = list(args.names)
            for notebook in args.tutorial:
                names += _tutorial_datasets(notebook)
            if args.all:
                names += [n for n, e in registry.DATASETS.items() if e.get("status") != "pending"]
            if not names:
                parser.error("name the datasets to fetch, or use --tutorial or --all")
            extras = "all" if args.extras == "all" else [x for x in args.extras.split(",") if x]
            for name in dict.fromkeys(names):
                fetch(name, extras=extras, data_dir=args.data_dir, force=args.force, verify=args.verify,
                      progress=not args.quiet, workers=args.workers)
        elif args.command == "verify":
            results = verify(*args.names, level="hash" if args.hash else "size", data_dir=args.data_dir)
            if not results:
                print("No datasets are downloaded yet.")
            for name, found in results.items():
                count = f"{len(found)} {'problem' if len(found) == 1 else 'problems'}"
                print(f"{name}: {count if found else 'OK'}")
                for problem in found[:20]:
                    print(f"  {problem}")
                if len(found) > 20:
                    print(f"  ... and {len(found) - 20} more")
            return 1 if any(results.values()) else 0
        elif args.command == "cache-key":
            print(cache_key(*args.names))
    except (DatasetNotAvailable, DownloadError, ValueError, ImportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped; run the same command again to continue", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
