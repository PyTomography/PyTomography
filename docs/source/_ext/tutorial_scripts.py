"""Plain-Python versions of the tutorial notebooks.

Each tutorial listed in ``docs/source/tutorials/tutorials.yaml`` is exported to
``examples/<section>/<NN>_<name>.py`` at the repository root: the code only, without
plots or explanations, with the notebook's headings kept as ``# %%`` cell markers so
editors such as VS Code and Spyder can run it cell by cell.

The scripts compute the tutorial's results and save them; only the notebooks plot (Luke,
13 Oct 2026). So the exporter leaves out every statement that plots (one that uses
matplotlib, PyTomography's plot_utils, or anything a plotting statement made, such as
``fig`` and ``ax``), notebook displays (a bare name on its own line), and then, until
nothing changes, every statement whose results only those left-out statements use: the
variables, imports and functions that exist for the plots. A statement any kept one needs
stays, even one that plots, so a script never loses a name it uses; a result nothing uses
afterwards stays too. Code cells whose metadata tags include ``plot`` (or ``skip-script``)
are left out whole. The docs build reads the same files to show a Script view on every
tutorial page, and ``docs/tools/export_scripts.py --check`` fails in CI when a script is out
of date or does not compile. ``from __future__`` imports are moved to the top of the script.

This module only needs the standard library and PyYAML, so it can run outside Sphinx.
"""
from __future__ import annotations

import ast
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


#: Modules whose names, once imported, mark a statement as plotting
PLOT_MODULES = ("matplotlib", "seaborn", "IPython.display", "ipywidgets", "pytomography.utils.plot_utils")
#: Names that plot or display in a notebook without an import
PLOT_BUILTINS = {"display", "plot_utils"}


def _is_plot_module(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in PLOT_MODULES)


def _plot_names(node: ast.AST) -> set[str]:
    """Names an import statement binds to a plotting module (or to something from one)."""
    out = set()
    if isinstance(node, ast.Import):
        for a in node.names:
            if _is_plot_module(a.name):
                out.add(a.asname or a.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        for a in node.names:
            if _is_plot_module(node.module) or _is_plot_module(f"{node.module}.{a.name}"):
                out.add(a.asname or a.name)
    return out


def _scope_names(node: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound, free names loaded) of a function, lambda, class or comprehension: its own locals stay inside."""
    bound, loads = set(), set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        bound |= {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        bound |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
        body = node.body if isinstance(node.body, list) else [node.body]
    elif isinstance(node, ast.ClassDef):
        body = node.body
    else:                                            # a comprehension
        body = [node]
    for b in body:
        for sub in ast.walk(b):
            if isinstance(sub, ast.Name):
                (bound if isinstance(sub.ctx, (ast.Store, ast.Del)) else loads).add(sub.id)
            elif isinstance(sub, ast.arg):
                bound.add(sub.arg)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and sub is not node:
                bound.add(sub.name)
    return bound, loads - bound


def _names(node: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    """(names a top-level statement binds, names it loads at its own level, free names of its nested scopes,
    names it deletes).

    A function's, lambda's, class's or comprehension's own names stay inside it; a loop's or with's targets are
    bound by the statement itself.
    """
    defs, loads, nested, deleted = set(), set(), set(), set()

    def visit(n):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(n.name)
            for d in list(getattr(n, "decorator_list", [])) + list(getattr(n, "bases", [])):
                visit(d)
            if not isinstance(n, ast.ClassDef):
                for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x is not None]:
                    visit(d)
            nested.update(_scope_names(n)[1])
            return
        if isinstance(n, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            nested.update(_scope_names(n)[1])
            return
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            defs.update(a.asname or a.name.split(".")[0] for a in n.names)
            return
        if isinstance(n, ast.Name):
            (defs if isinstance(n.ctx, ast.Store) else deleted if isinstance(n.ctx, ast.Del) else loads).add(n.id)
            return
        if isinstance(n, ast.Attribute) and n.attr == "plot_utils":
            loads.add("plot_utils")
        if isinstance(n, ast.ExceptHandler) and n.name:
            defs.add(n.name)
        for c in ast.iter_child_nodes(n):
            visit(c)

    visit(node)
    own = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.For, ast.AsyncFor)):
            own |= {x.id for x in ast.walk(sub.target) if isinstance(x, ast.Name)}
        elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
            own |= {x.id for x in ast.walk(sub.optional_vars) if isinstance(x, ast.Name)}
    return defs, loads - own, nested, deleted


_PLOT_CALL = re.compile(r"^(plot|imshow|show)(_\w+)?$")


def _calls_plot_function(node: ast.AST) -> bool:
    """A call of a function named plot..., imshow or show (PyTomography's plot_views, plot_arrow, ...)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if _PLOT_CALL.match(name):
                return True
    return False


def _is_display(node: ast.AST) -> bool:
    """A notebook display: an expression on its own that only names something (no call)."""
    return isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant))


def _definition_only(node: ast.AST) -> bool:
    """A statement whose only effect is to bind names: left out when only left-out statements use them."""
    return isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom,
                             ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


_COMPOUND = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try, ast.With, ast.AsyncWith)


def _prune(codes: list[str], skipped: list[bool] | None = None) -> list[str]:
    """The cells' code without plotting and what only the plots need (see the module docstring).

    ``skipped``: cells tagged to be left out; their statements start left out, and come back only when a kept
    statement needs a name one of them binds.
    """
    skipped = skipped or [False] * len(codes)
    stmts = []                       # (cell, node)
    parsed = []
    for i, code in enumerate(codes):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            parsed.append(None)      # kept as written (or left out, if tagged); the compile check reports it
            continue
        parsed.append(tree)
        stmts += [(i, node) for node in tree.body]
    n_st = len(stmts)
    info = [_names(n) for _, n in stmts]
    defs = [x[0] for x in info]
    loads = [x[1] for x in info]
    nested = [x[2] for x in info]
    deleted = [x[3] for x in info]

    def definers(k: int, name: str) -> list[int]:
        # the nearest statement before k that binds name, and the ones before it while those are loops or
        # branches (which may not bind it)
        out = []
        for j in range(k - 1, -1, -1):
            if name in defs[j]:
                out.append(j)
                if not isinstance(stmts[j][1], _COMPOUND):
                    break
        return out

    reach = []                       # per statement: the statements whose names it needs
    name_users: dict[tuple[int, str], set[int]] = {}   # (statement, name it binds): the statements that use it
    for k in range(n_st):
        r = set()
        for name in loads[k]:
            for j in definers(k, name):
                r.add(j)
                name_users.setdefault((j, name), set()).add(k)
        for name in nested[k]:       # a function's free names: whatever binds them, before or after
            for j in range(n_st):
                if j != k and name in defs[j]:
                    r.add(j)
                    name_users.setdefault((j, name), set()).add(k)
        reach.append(r)
    users = [set() for _ in range(n_st)]
    for k in range(n_st):
        for j in reach[k]:
            users[j].add(k)
    # plotting statements: plotting imports, displays, and statements that use what a plotting statement made (the
    # binding they see, so a name a plot cell reuses taints only the uses that see the plot's binding)
    plot = [bool(_plot_names(n) or _is_display(n) or _calls_plot_function(n) or (loads[k] | nested[k]) & PLOT_BUILTINS)
            for k, (_, n) in enumerate(stmts)]
    changed = True
    while changed:
        changed = False
        for k in range(n_st):
            if not plot[k] and any(plot[j] for j in reach[k]):
                plot[k] = True
                changed = True
    drop = [skipped[c] or plot[k] for k, (c, _) in enumerate(stmts)]
    # what only dropped statements use
    changed = True
    while changed:
        changed = False
        for k, (_, n) in enumerate(stmts):
            if not drop[k] and _definition_only(n) and users[k] and all(drop[j] for j in users[k]):
                drop[k] = True
                changed = True
    # never lose a name a kept statement uses: bring back what binds it
    changed = True
    while changed:
        changed = False
        for k in range(n_st):
            if not drop[k]:
                for j in reach[k]:
                    if drop[j]:
                        drop[j] = False
                        changed = True
    # a del (freeing memory) follows what it deletes: it goes when a name it deletes is no longer bound, and never
    # keeps a binding by itself
    for k, (_, n) in enumerate(stmts):
        if isinstance(n, ast.Delete) and not drop[k]:
            if any(not any(not drop[j] for j in definers(k, name)) for name in deleted[k]):
                drop[k] = True
    # cut the dropped statements' lines (and comment lines right above them) out of each cell, and from a kept
    # import the names only dropped statements use
    out = []
    for i, code in enumerate(codes):
        if parsed[i] is None:
            out.append("" if skipped[i] else code)
            continue
        lines = code.splitlines()
        cut = set()
        rewrite = {}                 # first line -> the import as it now reads
        for k, (c, n) in enumerate(stmts):
            if c != i:
                continue
            if not drop[k] and isinstance(n, (ast.Import, ast.ImportFrom)) and len(n.names) > 1:
                plot_only = {a for a in n.names
                             if name_users.get((k, a.asname or a.name.split(".")[0]))
                             and all(drop[j] for j in name_users[(k, a.asname or a.name.split(".")[0])])}
                if plot_only and len(plot_only) < len(n.names):
                    n2 = type(n)(**{**{f: getattr(n, f) for f in n._fields},
                                    "names": [a for a in n.names if a not in plot_only]})
                    rewrite[n.lineno - 1] = ast.unparse(n2)
                    cut.update(range(n.lineno, n.end_lineno))
                continue
            if not drop[k]:
                continue
            first = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            cut.update(range(first - 1, n.end_lineno))
            j = first - 2
            while j >= 0 and lines[j].lstrip().startswith("#"):
                cut.add(j)
                j -= 1
        kept = [rewrite.get(j, l) for j, l in enumerate(lines) if j not in cut]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n")
        out.append(text if any(l.strip() and not l.lstrip().startswith("#") for l in text.splitlines()) else "")
    return out


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


def _pop_future_imports(lines: list[str], future: list[str]) -> list[str]:
    """Move top-level ``from __future__ import ...`` statements out of ``lines`` into ``future``.

    Python only accepts them at the top of a file, and a notebook may have them in any cell
    (the GATE tutorials have them after the data cell), so the script puts them first.
    """
    kept = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("from __future__ import"):
            stmt = [lines[i]]
            if "(" in lines[i] and ")" not in lines[i]:  # parenthesised over several lines
                while i + 1 < len(lines) and ")" not in stmt[-1]:
                    i += 1
                    stmt.append(lines[i])
            stmt = "\n".join(stmt)
            if stmt not in future:
                future.append(stmt)
        else:
            kept.append(lines[i])
        i += 1
    return kept


def notebook_to_script(nb_path: Path, tut: dict, docs_url: str) -> str:
    nb = json.loads(Path(nb_path).read_text(encoding="utf8"))
    title = tut.get("title", Path(nb_path).stem)
    header = [
        '"""' + title,
        "",
        tut.get("summary", ""),
        "",
        f"Script version of the tutorial at {docs_url}",
        "It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.",
        "Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.",
        '"""',
    ]
    future: list[str] = []  # from __future__ imports, which must come first
    cells: list[tuple[str | None, str, bool]] = []   # (heading above it, code, tagged to leave out)
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
        lines = [l for l in src.rstrip().splitlines()
                 if not l.lstrip().startswith(("%", "!")) and l.strip() != "plt.show()"]
        lines = _pop_future_imports(lines, future)
        # a tagged cell is left out, but goes through the analysis: what it binds comes back if a kept cell needs it
        cells.append((pending_heading, "\n".join(lines).strip("\n"), bool(tags & SKIP_TAGS)))
        pending_heading = None
    body: list[str] = []
    heading = None
    for (h, _, _), code in zip(cells, _prune([c for _, c, _ in cells], [t for _, _, t in cells])):
        heading = h or heading                  # a heading whose code is all left out moves to the next code
        if not code.strip():
            continue
        if heading:
            body.append(f"# %% {heading}")
            heading = None
        body.append(code)
        body.append("")
    # a statement that plots stays only when a kept one needs what it makes: then no figure windows
    plots = any(re.search(r"\b(plt|matplotlib|plot_utils)\b", b) for b in body)
    preamble = ["import matplotlib", 'matplotlib.use("Agg")  # no figure windows when run as a script', ""] if plots else []
    return "\n".join(header + future + preamble + body).rstrip() + "\n"


def compile_errors(srcdir: Path, docs_base: str = "https://pytomography.readthedocs.io/en/latest/") -> list[str]:
    """Compile every script as the exporter writes it. Returns one message per script that does not compile."""
    srcdir = Path(srcdir)
    paths = script_paths(srcdir)
    errors = []
    for section in load_sections(srcdir):
        for tut in section["tutorials"]:
            nb_path = srcdir / "notebooks" / f"{tut['notebook']}.ipynb"
            text = notebook_to_script(nb_path, tut, f"{docs_base}notebooks/{tut['notebook']}.html")
            rel = paths[f"notebooks/{tut['notebook']}"].as_posix()
            try:
                compile(text, rel, "exec")
            except SyntaxError as e:
                errors.append(f"{rel}:{e.lineno}: {e.msg}")
    return errors


def readme(srcdir: Path, docs_base: str) -> str:
    lines = [
        "# PyTomography example scripts",
        "",
        "Plain-Python versions of every tutorial: the code that computes and saves the results, without the plots,",
        "what only the plots need, or the explanations, so you can read a whole pipeline at a glance or run it as a",
        "script. Each file links to the full tutorial on the docs site.",
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
