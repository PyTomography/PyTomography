"""Site-specific Sphinx helpers for the PyTomography docs.

* ``tutorials/tutorials.yaml`` is the single list of tutorials. From it this extension
  builds the filterable gallery (``.. tutorial-gallery::``), the sidebar toctree, and
  a thumbnail for every tutorial taken from the notebook's own image outputs.
* ``tutorial_data.py`` holds everything about the tutorial data, from the registry that
  ``pytomography.datasets.fetch()`` downloads from: the "Tutorial data" page
  (``.. tutorial-datasets::``) and each tutorial's datasets for its header.
* Every notebook page gets a header (``tutorial_page.py``): its summary and tags, a panel with
  Run it (Colab, GitHub), Data (a popup per dataset) and Results (View results in 3D), and tabs
  for the Jupyter notebook or the script.
* Every page is also written as Markdown to ``_md/<page>.md``, and ``llms.txt`` and
  ``llms-full.txt`` are built from those copies for AI agents. A ``pt-source`` meta tag
  points "Copy page as Markdown" at the copy.
"""
from __future__ import annotations

import base64
import html
import json
import posixpath
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx import addnodes

import tutorial_page
import tutorial_scripts
from tutorial_data import dataset_anchor

try:                      # the 3D viewer's images, for the header's View results in 3D
    import pytomo_viewer
except ImportError:
    pytomo_viewer = None

REPO = "PyTomography/PyTomography"
BRANCH = "main"
NOTEBOOK_DIR = "docs/source/notebooks"
THUMB_DIR = "_generated/thumbs"  # relative to srcdir, listed in html_static_path


def _load_tutorials(srcdir: Path) -> list[dict]:
    with open(srcdir / "tutorials" / "tutorials.yaml", encoding="utf8") as f:
        return yaml.safe_load(f)["sections"]


def _write_thumbnails(app) -> None:
    """Save one image output per tutorial notebook as its gallery thumbnail."""
    srcdir = Path(app.srcdir)
    out = srcdir / THUMB_DIR
    out.mkdir(parents=True, exist_ok=True)
    for section in app.env.pytomo_tutorials:
        for tut in section["tutorials"]:
            nb_path = srcdir / "notebooks" / f"{tut['notebook']}.ipynb"
            target = out / f"{tut['notebook']}.png"
            if target.exists() and target.stat().st_mtime >= nb_path.stat().st_mtime:
                continue  # unchanged, and rewriting would retrigger sphinx-autobuild
            nb = json.loads(nb_path.read_text(encoding="utf8"))
            images = [o["data"]["image/png"] for c in nb["cells"]
                      for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
            if not images:
                continue
            png = images[tut.get("thumbnail", -1)]
            if isinstance(png, list):
                png = "".join(png)
            target.write_bytes(base64.b64decode(png))


def builder_inited(app):
    app.env.pytomo_tutorials = _load_tutorials(Path(app.srcdir))
    app.env.pytomo_by_notebook = {
        f"notebooks/{t['notebook']}": t
        for s in app.env.pytomo_tutorials for t in s["tutorials"]
    }
    _write_thumbnails(app)


class TutorialGallery(Directive):
    """Filterable card gallery of every tutorial, plus the hidden toctrees."""

    has_content = False

    def run(self):
        env = self.state.document.settings.env
        env.note_dependency(str(Path(env.srcdir) / "tutorials" / "tutorials.yaml"))
        here = posixpath.dirname(env.docname)
        sections = env.pytomo_tutorials
        rel = lambda target: posixpath.relpath(target, here or ".")
        thumbs = Path(env.srcdir) / THUMB_DIR

        filters = {
            "modality": ["SPECT", "PET", "CT"],
            "data": sorted({t["data"] for s in sections for t in s["tutorials"]} - {"None"}),
            "topic": sorted({t["topic"] for s in sections for t in s["tutorials"]}),
        }
        parts = ['<div class="pt-gallery" data-gallery>', '<div class="pt-filters" role="group" aria-label="Filter tutorials">']
        for key, values in filters.items():
            parts.append(f'<div class="pt-fgroup"><span>{key.title()}</span>')
            parts.append(f'<button type="button" class="pt-chip" data-key="{key}" data-value="All" aria-pressed="true">All</button>')
            for v in values:
                parts.append(f'<button type="button" class="pt-chip" data-key="{key}" data-value="{html.escape(v)}" aria-pressed="false">{html.escape(v)}</button>')
            parts.append("</div>")
        parts.append('<span class="pt-count" aria-live="polite"></span></div>')

        for section in sections:
            parts.append(f'<section class="pt-gsection"><h2 class="pt-gtitle">{html.escape(section["title"])}</h2><div class="pt-cards">')
            for t in section["tutorials"]:
                href = rel(f"notebooks/{t['notebook']}") + ".html"
                has_thumb = (thumbs / f"{t['notebook']}.png").exists()
                thumb = (f'<img src="{rel("_static/thumbs/" + t["notebook"] + ".png")}" alt="" loading="lazy">'
                         if has_thumb else '<span class="pt-thumb-empty"></span>')
                parts.append(
                    f'<a class="pt-card" href="{href}" data-modality="{t["modality"]}" data-data="{t["data"]}" data-topic="{t["topic"]}">'
                    f'<span class="pt-thumb">{thumb}<span class="pt-mod">{t["modality"] if t["modality"] != "Any" else "All modalities"}</span></span>'
                    f'<span class="pt-cbody"><span class="pt-ctitle">{html.escape(t["title"])}</span>'
                    f'<span class="pt-csum">{html.escape(t["summary"])}</span>'
                    f'<span class="pt-tags"><span>{t["data"] if t["data"] != "None" else "No data needed"}</span><span>{t["topic"]}</span></span></span></a>'
                )
            parts.append("</div></section>")
        parts.append('<p class="pt-empty" hidden>No tutorial matches all three filters yet. <a href="../contributing/feature-board.html">Write one?</a></p></div>')

        result = [nodes.raw("", "".join(parts), format="html")]
        # Hidden toctrees so every tutorial appears in the sidebar under its section.
        for section in sections:
            tocnode = addnodes.toctree()
            tocnode["parent"] = env.docname
            entries = [(None, f"notebooks/{t['notebook']}") for t in section["tutorials"]]
            tocnode["entries"] = entries
            tocnode["includefiles"] = [e[1] for e in entries]
            tocnode["maxdepth"] = 1
            tocnode["caption"] = section["title"]
            tocnode["glob"] = False
            tocnode["hidden"] = True
            tocnode["includehidden"] = False
            tocnode["numbered"] = 0
            tocnode["titlesonly"] = True
            wrapper = nodes.compound(classes=["toctree-wrapper"])
            wrapper += tocnode
            result.append(wrapper)
        return result


class FeatureBoard(Directive):
    """Cards for contributing/feature_board.yaml, grouped by difficulty, filterable by area."""

    has_content = False
    LEVELS = [
        ("good-first", "Good first issues", "An afternoon to a few days. The fix location is known."),
        ("intermediate", "Intermediate", "One to three weeks. One module, with a clear test."),
        ("advanced", "Advanced and research", "A month or more. A new feature or study, with a mentor."),
    ]
    LABELS = {"good-first": "Good first issue", "intermediate": "Intermediate",
              "advanced": "Advanced", "research": "Research-sized"}

    def run(self):
        env = self.state.document.settings.env
        path = Path(env.srcdir) / "contributing" / "feature_board.yaml"
        env.note_dependency(str(path))
        items = yaml.safe_load(path.read_text(encoding="utf8"))["items"]
        areas = sorted({i["area"] for i in items})
        esc = lambda v: html.escape(str(v))
        parts = ['<div class="pt-board" data-board><div class="pt-filters" role="group" aria-label="Filter by area">',
                 '<div class="pt-fgroup"><span>Area</span>',
                 '<button type="button" class="pt-chip" data-key="area" data-value="All" aria-pressed="true">All</button>']
        parts += [f'<button type="button" class="pt-chip" data-key="area" data-value="{esc(a)}" aria-pressed="false">{esc(a)}</button>' for a in areas]
        parts.append('</div><span class="pt-count" aria-live="polite"></span></div><div class="pt-cols">')
        for key, title, desc in self.LEVELS:
            col = [i for i in items if (i["level"] in ("advanced", "research")) == (key == "advanced") and (key == "advanced" or i["level"] == key)]
            parts.append(f'<div class="pt-col"><div class="pt-colhead"><b>{title}</b><span>{len(col)}</span></div><p>{desc}</p>')
            for i in col:
                issue = (f'<a href="https://github.com/{REPO}/issues/{i["issue"]}">#{i["issue"]}</a>'
                         if str(i["issue"]).isdigit() else "<span>issue to open</span>")
                parts.append(
                    f'<article class="pt-bcard" data-area="{esc(i["area"])}"><h3>{esc(i["title"])}</h3>'
                    f'<div class="pt-tags"><span class="pt-lvl-{i["level"]}">{self.LABELS[i["level"]]}</span><span>{esc(i["area"])}</span>{issue}</div>'
                    f'<p>{esc(i["why"])}</p><dl><dt>Data</dt><dd>{esc(i["data"])}</dd><dt>Start</dt><dd>{esc(i["start"])}</dd>'
                    f'<dt>Done</dt><dd>{esc(i["done"])}</dd></dl></article>'
                )
            parts.append("</div>")
        parts.append("</div></div>")
        return [nodes.raw("", "".join(parts), format="html")]


def add_notebook_header(app, doctree):
    """Insert the header under the title of every tutorial notebook page (tutorial_page.header_html)."""
    docname = app.env.docname
    if not app.env.doc2path(docname).suffix == ".ipynb":
        return
    nb = Path(docname).name
    t = app.env.pytomo_by_notebook.get(docname, {})
    colab = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{NOTEBOOK_DIR}/{nb}.ipynb"
    github = f"https://github.com/{REPO}/blob/{BRANCH}/{NOTEBOOK_DIR}/{nb}.ipynb"
    # Written by docs/tools/run_tutorials.py --write-back when the notebook last ran end to end
    run = json.loads(app.env.doc2path(docname).read_text(encoding="utf8")).get("metadata", {}).get("pytomography_run")
    # The plain-Python version of this tutorial, generated into examples/ by docs/tools/export_scripts.py
    script_rel = tutorial_scripts.script_paths(Path(app.srcdir)).get(docname)
    script_file = tutorial_scripts.repo_root(Path(app.srcdir)) / script_rel if script_rel else None
    has_script = script_file is not None and script_file.exists()
    bar = tutorial_page.header_html(
        t, colab=colab, github=github, has_script=has_script, run=run,
        data=tutorial_page.page_data(app.env, docname), data_page=tutorial_page.data_page_href(docname),
        viewer=pytomo_viewer.for_page(app.env, docname) if pytomo_viewer else None)
    section = next(iter(doctree.findall(nodes.section)), None)
    if section is None:
        return
    title_index = next((i for i, n in enumerate(section.children) if isinstance(n, nodes.title)), -1)
    section.insert(title_index + 1, nodes.raw("", bar, format="html"))
    tutorial_page.drop_data_note(section)
    if has_script:
        app.env.note_dependency(str(script_file))
        script_url = f"https://github.com/{REPO}/blob/{BRANCH}/{script_rel.as_posix()}"
        view = nodes.container(classes=["pt-script-view"])
        view += nodes.raw("", (
            '<p class="pt-script-note">The code of this tutorial without plots or explanations. '
            'Run it as a file, or cell by cell in an editor that understands <code># %%</code> markers. '
            f'<a href="{script_url}">{html.escape(script_rel.as_posix())} on GitHub</a></p>'), format="html")
        code = script_file.read_text(encoding="utf8")
        block = nodes.literal_block(code, code, language="python")
        view += block
        section.insert(title_index + 2, view)


def add_source_meta(app, pagename, templatename, context, doctree):
    if pagename in app.env.found_docs and "pathto" in context:
        url = context["pathto"](f"_md/{pagename}.md", 1)
        context["metatags"] = context.get("metatags", "") + f'\n<meta name="pt-source" content="{url}">'


# -- Markdown copies of every page, llms.txt and llms-full.txt ----------------

LLMS_SUMMARY = (
    "PyTomography is an open-source Python library for quantitative SPECT, PET and CT "
    "reconstruction on the GPU. Each link below is a Markdown version of one documentation page."
)


def _notebook_to_markdown(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf8"))
    out = []
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "markdown":
            out.append(src)
        elif cell["cell_type"] == "code":
            out.append(f"```python\n{src}\n```")
            for o in cell.get("outputs", []):
                data = o.get("data", {})
                if "image/png" in data or "image/jpeg" in data:
                    out.append("*[figure output]*")
                    continue
                text = o.get("text") or data.get("text/plain")
                if text:
                    text = "".join(text) if isinstance(text, list) else text
                    lines = text.rstrip().splitlines()
                    if len(lines) > 15:
                        lines = lines[:15] + [f"... ({len(lines) - 15} more lines)"]
                    out.append("Output:\n```text\n" + "\n".join(lines) + "\n```")
    return "\n\n".join(out) + "\n"


def _page_markdown(env, docname: str) -> str:
    path = Path(env.doc2path(docname))
    if path.suffix == ".ipynb":
        return _notebook_to_markdown(path)
    text = path.read_text(encoding="utf8")
    if text.startswith("---"):  # drop MyST front matter
        end = text.find("\n---", 3)
        text = text[end + 4:].lstrip() if end > 0 else text
    return text


def _toctree_order(env) -> list[str]:
    seen, order = set(), []

    def visit(doc):
        if doc in seen or doc not in env.found_docs:
            return
        seen.add(doc)
        order.append(doc)
        for child in env.toctree_includes.get(doc, []):
            visit(child)

    visit(env.config.root_doc)
    return order + sorted(d for d in env.found_docs if d not in seen)


def write_markdown(app, exception):
    if exception or app.builder.format != "html":
        return
    env, outdir = app.env, Path(app.outdir)
    title = lambda d: env.titles[d].astext() if d in env.titles else d
    groups = {"Guides": [], "Tutorials": [], "API reference": []}
    full = [f"# PyTomography documentation\n\n> {LLMS_SUMMARY}\n"]
    for doc in _toctree_order(env):
        if doc in ("genindex", "search", "py-modindex"):
            continue
        md = _page_markdown(env, doc)
        target = outdir / "_md" / f"{doc}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"<!-- {title(doc)} -->\n\n" + md, encoding="utf8")
        group = ("API reference" if doc.startswith("api/") else
                 "Tutorials" if doc.startswith("notebooks/") else "Guides")
        groups[group].append(f"- [{title(doc)}](_md/{doc}.md)")
        full.append(f"\n\n---\n\n<!-- page: {doc} -->\n\n{md}")
    index = [f"# PyTomography\n\n> {LLMS_SUMMARY}\n"]
    for name, links in groups.items():
        index.append(f"\n## {name}\n\n" + "\n".join(links) + "\n")
    (outdir / "llms.txt").write_text("".join(index), encoding="utf8")
    (outdir / "llms-full.txt").write_text("".join(full), encoding="utf8")


def setup(app):
    app.setup_extension("tutorial_data")  # the Tutorial data page and each tutorial's datasets
    app.add_directive("tutorial-gallery", TutorialGallery)
    app.add_directive("feature-board", FeatureBoard)
    app.connect("builder-inited", builder_inited)
    app.connect("doctree-read", add_notebook_header)
    app.connect("html-page-context", add_source_meta)
    app.connect("build-finished", write_markdown, priority=100)  # before autoapi removes its sources
    return {"parallel_read_safe": True, "parallel_write_safe": True}
