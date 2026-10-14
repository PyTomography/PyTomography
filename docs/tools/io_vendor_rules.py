"""The vendor rules shown on the Architecture page's "Data and conventions" tab (``vendor_rules`` in
``docs/source/_static/arch/io-conventions.json``) against the library's vendor table, ``src/pytomography/io/_vendors.py``.

The library's table is read with ``ast``, without importing PyTomography::

    python docs/tools/io_vendor_rules.py            # say whether the two agree (exit 1 if not)
    python docs/tools/io_vendor_rules.py --write    # copy the library's table into the JSON

The docs build (``_ext/pytomo_io.py``) runs the same comparison and warns when they differ.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
JSON_PATH = DOCS / "source" / "_static" / "arch" / "io-conventions.json"
VENDORS_PATH = DOCS.parent / "src" / "pytomography" / "io" / "_vendors.py"
FIELDS = ("id", "vendor", "data", "rule", "evidence", "status")


def library_rules(path: Path = VENDORS_PATH) -> list[dict] | None:
    """The rules of ``_vendors.py`` (its ``_RULES`` list of ``VendorRule(...)``), in order; None when there is no such
    file (a library from before the vendor table)."""
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf8"))
    for node in tree.body:
        targets = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        if "_RULES" in targets and isinstance(node.value, ast.List):
            rules = []
            for call in node.value.elts:
                if not (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "VendorRule"):
                    raise ValueError(f"{path}: _RULES holds something other than VendorRule(...)")
                values = [ast.literal_eval(a) for a in call.args]
                values += [ast.literal_eval(k.value) for k in call.keywords]
                rules.append(dict(zip(FIELDS, values)))
            return rules
    raise ValueError(f"{path}: no _RULES list")


def differences(page: list[dict], library: list[dict]) -> list[str]:
    """What differs between the page's rules and the library's, one line each."""
    out = []
    by_id = {r["id"]: r for r in page}
    lib_ids = [r["id"] for r in library]
    for r in library:
        mine = by_id.get(r["id"])
        if mine is None:
            out.append(f"{r['id']}: in the library, not on the page")
            continue
        for f in FIELDS[1:]:
            if mine.get(f) != r[f]:
                out.append(f"{r['id']}: {f} differs")
    out += [f"{i}: on the page, not in the library" for i in by_id if i not in lib_ids]
    if not out and [r["id"] for r in page] != lib_ids:
        out.append("the rules are in another order")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="copy the library's table into io-conventions.json")
    args = parser.parse_args()
    library = library_rules()
    if library is None:
        print(f"{VENDORS_PATH} does not exist: nothing to compare")
        return 0
    data = json.loads(JSON_PATH.read_text(encoding="utf8"))
    diff = differences(data["vendor_rules"], library)
    if args.write:
        data["vendor_rules"] = library
        JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf8")
        print(f"wrote {len(library)} rules to {JSON_PATH}")
        return 0
    if diff:
        print("io-conventions.json and _vendors.py differ:\n  " + "\n  ".join(diff))
        return 1
    print(f"the {len(library)} rules agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
