"""Export every tutorial notebook to a plain Python script in examples/.

    python docs/tools/export_scripts.py              # write examples/**/*.py and examples/README.md
    python docs/tools/export_scripts.py --check      # exit 1 if any script is out of date or does not compile (for CI)
    python docs/tools/export_scripts.py --tag-plots  # first tag plotting-only cells with "plot"

The tutorials and their order come from docs/source/tutorials/tutorials.yaml.
"""
import argparse
import sys
from pathlib import Path

SRCDIR = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRCDIR / "_ext"))

from tutorial_scripts import compile_errors, export_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="only report scripts that are out of date or do not compile")
    parser.add_argument("--tag-plots", action="store_true", help="tag plotting-only cells before exporting")
    args = parser.parse_args()
    changed = export_all(SRCDIR, check=args.check, tag_plots=args.tag_plots)
    errors = compile_errors(SRCDIR)
    if args.check:
        if changed:
            print("Out of date (run python docs/tools/export_scripts.py):")
            print("\n".join(f"  {c}" for c in changed))
    else:
        print(f"Wrote {len(changed)} file(s)" + (":\n" + "\n".join(f"  {c}" for c in changed) if changed else "."))
    if errors:
        print("Do not compile (fix the notebook, or the exporter in docs/source/_ext/tutorial_scripts.py):")
        print("\n".join(f"  {e}" for e in errors))
        return 1
    if args.check:
        if changed:
            return 1
        print("All example scripts match their notebooks and compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
