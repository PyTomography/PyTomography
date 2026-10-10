"""Plain-Python versions of the tutorial notebooks.

Each tutorial listed in ``docs/source/tutorials/tutorials.yaml`` is exported to
``examples/<section>/<NN>_<name>.py`` at the repository root: the code only, without
plots or explanations, with the notebook's headings kept as ``# %%`` cell markers so
editors such as VS Code and Spyder can run it cell by cell.

Code cells whose metadata tags include ``plot`` (or ``skip-script``) are left out. The
docs build reads the same files to show a Script view on every tutorial page, and
``docs/tools/export_scripts.py --check`` fails in CI when a script is out of date.

This module only needs the standard library and PyYAML, so it can run outside Sphinx.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

SKIP_TAGS = {"plot", "skip-script"}
# A code cell counts as plotting when most of its lines call matplotlib or PyTomography's plotting helpers
_PLOT_LINE = re.compile(
    r"\bplt\.|\bfig\b|\bax(es)?\b|\.imshow\(|\.pcolormesh\(|\.colorbar\(|\.legend\(|\.set_(title|xlabel|ylabel|xlim|ylim)\(|"
    r"\bsubplots?\(|dual_imshow|\.axis\(|\.savefig\("
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def repo_root(srcdir: Path) -> Path:
    """``docs/source`` -> repository root."""
    return Path(srcdir).resolve().parents[1]


def load_sections(srcdir: Path) -> list[dict]:
    with open(Path(srcdir) / "tutorials" / "tutorials.yaml", encoding="utf8") as f:
        return yaml.safe_load(f)["sections"]


def script_paths(srcdir: Path) -> dict[str, Path]:
    """Map ``notebooks/<name>`` docnames to their script path, relative to the repository root."""
    out = {}
    for section in load_sections(srcdir):
        folder = section.get("folder") or slug(section["title"])
        for i, tut in enumerate(section["tutorials"], start=1):
            out[f"notebooks/{tut['notebook']}"] = Path("examples") / folder / f"{i:02d}_{slug(tut['title'])}.py"
    return out


def is_plot_cell(source: str) -> bool:
    lines = [l for l in source.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return False
    hits = sum(bool(_PLOT_LINE.search(l)) for l in lines)
    return hits >= 0.6 * len(lines)


def notebook_to_script(nb_path: Path, tut: dict, docs_url: str) -> str:
    nb = json.loads(Path(nb_path).read_text(encoding="utf8"))
    title = tut.get("title", Path(nb_path).stem)
    header = [
        '"""' + title,
        "",
        tut.get("summary", ""),
        "",
        f"Script version of the tutorial at {docs_url}",
        "It keeps the computation and leaves out the plots and explanations.",
        "Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.",
        '"""',
        "import matplotlib",
        'matplotlib.use("Agg")  # no figure windows when run as a script',
        "",
    ]
    body: list[str] = []
    pending_heading: str | None = None
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "markdown":
            for line in src.splitlines():
                if line.startswith("#") and not line.startswith("#!"):
                    text = line.lstrip("#").strip()
                    if text and text.lower() != title.lower() and text.lower() != Path(nb_path).stem.lower():
                        pending_heading = text
            continue
        if cell["cell_type"] != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []))
        if tags & SKIP_TAGS:
            continue
        lines = [l for l in src.rstrip().splitlines()
                 if not l.lstrip().startswith(("%", "!")) and l.strip() != "plt.show()"]
        code = "\n".join(lines).strip("\n")
        if not code.strip():
            continue
        if pending_heading:
            body.append(f"# %% {pending_heading}")
            pending_heading = None
        body.append(code)
        body.append("")
    return "\n".join(header + body).rstrip() + "\n"


def readme(srcdir: Path, docs_base: str) -> str:
    lines = [
        "# PyTomography example scripts",
        "",
        "Plain-Python versions of every tutorial: the code only, without plots or explanations, so you can read",
        "a whole pipeline at a glance or run it as a script. Each file links to the full tutorial on the docs site.",
        "",
        "These files are generated from the notebooks in `docs/source/notebooks` by `docs/tools/export_scripts.py`.",
        "Edit the notebook, then run that script; CI checks the two match.",
        "",
        "Each script downloads the data it reads with `pytomography.datasets.fetch()`; the Tutorial data page of the",
        "docs lists every dataset.",
        "",
    ]
    paths = script_paths(srcdir)
    for section in load_sections(srcdir):
        lines += [f"## {section['title']}", "", "| Script | What it does |", "|---|---|"]
        for tut in section["tutorials"]:
            p = paths[f"notebooks/{tut['notebook']}"]
            rel = p.relative_to("examples").as_posix()
            lines.append(f"| [`{rel}`]({rel}) | {tut['summary']} [Tutorial]({docs_base}notebooks/{tut['notebook']}.html) |")
        lines.append("")
    return "\n".join(lines)


def export_all(srcdir: Path, docs_base: str = "https://pytomography.readthedocs.io/en/latest/",
               check: bool = False, tag_plots: bool = False) -> list[str]:
    """Write (or with ``check=True`` compare) every script and the README. Returns the paths that changed."""
    srcdir = Path(srcdir)
    root = repo_root(srcdir)
    changed = []
    paths = script_paths(srcdir)
    for section in load_sections(srcdir):
        for tut in section["tutorials"]:
            nb_path = srcdir / "notebooks" / f"{tut['notebook']}.ipynb"
            if tag_plots:
                tag_plot_cells(nb_path)
            text = notebook_to_script(nb_path, tut, f"{docs_base}notebooks/{tut['notebook']}.html")
            target = root / paths[f"notebooks/{tut['notebook']}"]
            if not target.exists() or target.read_text(encoding="utf8") != text:
                changed.append(target.relative_to(root).as_posix())
                if not check:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf8", newline="\n")
    # Scripts left behind when a tutorial is renamed, renumbered or removed
    expected = {(root / p).resolve() for p in paths.values()}
    for stale in sorted((root / "examples").glob("*/*.py")):
        if stale.resolve() not in expected:
            changed.append(stale.relative_to(root).as_posix() + " (stale)")
            if not check:
                stale.unlink()
    target = root / "examples" / "README.md"
    text = readme(srcdir, docs_base)
    if not target.exists() or target.read_text(encoding="utf8") != text:
        changed.append("examples/README.md")
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf8", newline="\n")
    return changed


def tag_plot_cells(nb_path: Path) -> int:
    """Add a ``plot`` tag to code cells that only make figures. Returns how many were tagged."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf8"))
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        tags = cell.get("metadata", {}).get("tags", [])
        if "plot" not in tags and is_plot_cell(src):
            cell.setdefault("metadata", {})["tags"] = tags + ["plot"]
            n += 1
    if n:
        # Same layout Jupyter writes (indent 1, sorted keys), so the diff only shows the new tags
        text = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        Path(nb_path).write_text(text, encoding="utf8", newline="\n")
    return n
