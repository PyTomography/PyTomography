"""The tutorial data, for the docs: what each dataset is, where it comes from, its licence, citation and size, and
the ``datasets.fetch()`` line that downloads it. Everything comes from ``src/pytomography/datasets/registry.py``,
the registry that ``pytomography.datasets.fetch()`` downloads from, so the docs and the code never disagree.

For the rest of the site:

* ``for_page(env, docname)``: the datasets a tutorial page reads, as dicts with the keys of ``entry()`` plus
  ``used_by``, the other tutorials that read the same dataset ([{"notebook", "title"}]), in the order
  ``tutorials.yaml`` lists them; ``[]`` for a page that reads none. The tutorial header's data popup uses it.
* ``page_json(env, docname)``: the same, as JSON that is safe inside ``<script type="application/json">``.
* ``dataset_anchor(key)``: the id of a dataset's entry on the Tutorial data page (also an ``entry()`` field).
* ``.. tutorial-datasets::``: the entries of the Tutorial data page.
"""
from __future__ import annotations

import html
import json
import posixpath
import runpy
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)


def registry_path(srcdir) -> Path:
    return Path(srcdir).parents[1] / "src" / "pytomography" / "datasets" / "registry.py"


def _load_registry(srcdir) -> dict:
    """The registry imports nothing, so it loads without importing PyTomography."""
    path = registry_path(srcdir)
    if not path.is_file():
        raise ExtensionError(f"tutorial_data: the dataset registry {path} is missing; the tutorials' data and the"
                             " Tutorial data page are built from it")
    return runpy.run_path(str(path))["DATASETS"]


def _size_text(n: float) -> str:
    """Bytes in decimal units, as pytomography.datasets prints them."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.{2 if unit == 'GB' and n < 1e11 else 1}f} {unit}"
    return f"{int(n)} B"


def _sizes(d: dict) -> tuple[int, int]:
    """(bytes downloaded, bytes on disk) of a registry entry, counted as pytomography.datasets counts them."""
    download = disk = 0
    for p in d["parts"]:
        if p["kind"] == "zip_range":
            download += p["size"] - p["index"][0] + sum(hi - lo for lo, hi, _ in p["ranges"])
        elif p["kind"] != "package":
            download += p["size"]
        disk += p["unpacked"] if p["kind"] in ("zip", "zip_range") else p["size"]
    return download, disk


def dataset_anchor(key: str) -> str:
    return "data-" + "".join(c if c.isalnum() else "-" for c in key.lower())


def entry(key: str, d: dict) -> dict:
    """What the docs show about one dataset. These keys are the interface the tutorial header relies on."""
    available = d.get("status") != "pending"
    download, disk = _sizes(d)
    return {
        "key": key,
        "title": d["title"],                  # what it is
        "source": d["source"],                # who published it
        "url": d.get("url") or "",            # its landing page, a DOI when there is one
        "licence": d["licence"],
        "cite": d.get("cite") or "",
        "download": _size_text(download) if available else "",
        "disk": _size_text(disk) if available else "",
        "download_bytes": download if available else 0,
        "fetch": f'datasets.fetch("{key}")',
        "needs": "pip install idc-index" if any(p["kind"] == "idc" for p in d["parts"]) else "",
        "available": available,
        "note": d.get("note") or "",          # why it cannot be downloaded yet
        "anchor": dataset_anchor(key),
    }


def builder_inited(app):
    registry = _load_registry(app.srcdir)
    with open(Path(app.srcdir) / "tutorials" / "tutorials.yaml", encoding="utf8") as f:
        tutorials = [t for s in yaml.safe_load(f)["sections"] for t in s["tutorials"]]
    app.env.pytomo_data = {key: entry(key, d) for key, d in registry.items()}
    app.env.pytomo_data_by_notebook = {t["notebook"]: list(t.get("datasets") or []) for t in tutorials}
    app.env.pytomo_data_titles = {t["notebook"]: t["title"] for t in tutorials}
    for notebook, keys in app.env.pytomo_data_by_notebook.items():
        for key in keys:
            if key not in registry:
                logger.warning(f"tutorials.yaml: {notebook} reads {key}, which is not in the dataset registry")


def for_page(env, docname: str) -> list[dict]:
    """The datasets a page reads, in tutorials.yaml's order, each with ``used_by``: the other tutorials that read it.
    [] for a page that is not a tutorial or reads no data."""
    if not docname.startswith("notebooks/"):
        return []
    notebook = docname[len("notebooks/"):]
    out = []
    for key in env.pytomo_data_by_notebook.get(notebook, []):
        if key in env.pytomo_data:
            used_by = [{"notebook": nb, "title": env.pytomo_data_titles[nb]}
                       for nb, keys in env.pytomo_data_by_notebook.items() if key in keys and nb != notebook]
            out.append(dict(env.pytomo_data[key], used_by=used_by))
    return out


def page_json(env, docname: str) -> str:
    """for_page() as JSON for a <script type="application/json"> element: "</" is escaped so it cannot end it."""
    return json.dumps(for_page(env, docname)).replace("</", "<\\/")


class TutorialDatasets(Directive):
    """One entry per dataset in the registry: the line that downloads it, its size, licence and citation, and the
    tutorials that use it."""

    has_content = False

    def run(self):
        env = self.state.document.settings.env
        env.note_dependency(str(registry_path(env.srcdir)))
        here = posixpath.dirname(env.docname)
        esc = lambda v: html.escape(str(v))
        parts = ['<div class="pt-datasets">']
        for key, e in env.pytomo_data.items():
            used = ", ".join(
                f'<a href="{posixpath.relpath("notebooks/" + nb, here)}.html">{esc(env.pytomo_data_titles[nb])}</a>'
                for nb, keys in env.pytomo_data_by_notebook.items() if key in keys) or "not used by a tutorial yet"
            if not e["available"]:
                rows = [("Get it", f"Not downloadable yet. {esc(e['note'])}")]
            else:
                get = f"<code>{esc(e['fetch'])}</code>"
                if e["needs"]:
                    get += f", which needs the Imaging Data Commons client: <code>{esc(e['needs'])}</code>"
                rows = [("Get it", get), ("Size", f"{e['download']} to download, {e['disk']} on disk")]
            rows += [("Licence", esc(e["licence"])), ("Used by", used)]
            if e["cite"]:
                rows.append(("Cite", esc(e["cite"])))
            parts.append(
                f'<article class="pt-dataset" id="{e["anchor"]}"><h3><code>{esc(key)}</code></h3>'
                f'<p>{esc(e["title"])}. <a href="{esc(e["url"])}">{esc(e["source"])}</a></p><dl>'
                + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl></article>")
        parts.append("</div>")
        return [nodes.raw("", "".join(parts), format="html")]


def setup(app):
    app.add_directive("tutorial-datasets", TutorialDatasets)
    app.connect("builder-inited", builder_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
