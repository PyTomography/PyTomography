"""The header of every tutorial page: one convention for all of them (Luke, 9 Oct 2026).

Under the page's title, in this order:

* the tutorial's one-line summary, its tags, and when it last ran;
* one panel with a labelled column per job: **Run it** (Open in Colab, View on GitHub), **Data** (a button per
  dataset, which opens a popup saying what it is, where it comes from, its licence and citation, its size and how to
  get it) and **Results** (View results in 3D, which opens the 3D viewer full screen);
* tabs for the page itself, "Jupyter notebook (.ipynb)" and "Script (.py)", with Copy page as Markdown on the right.

``header_html()`` builds it; ``pytomo_docs.add_notebook_header`` puts it under the title. The popup is drawn by
``js/pytomo-tutorial.js`` from the page's datasets, embedded as JSON: ``tutorial_data.for_page()`` (the registry of
``pytomography.datasets``) when that module is present, else the entries of ``tutorials/datasets.yaml``. The
notebook's own "Data." note is dropped from the page (``drop_data_note``): the popup says the same.
"""
from __future__ import annotations

import html
import json
import posixpath

from docutils import nodes

_PATHS = {
    "play": '<path d="M6.5 4.8v10.4a.6.6 0 0 0 .9.5l8.3-5.2a.6.6 0 0 0 0-1L7.4 4.3a.6.6 0 0 0-.9.5z" fill="currentColor" stroke="none"/>',
    "code": '<path d="M7.5 6 3.5 10l4 4M12.5 6l4 4-4 4"/>',
    "cube": '<path d="M10 2.6 16.4 6v8L10 17.4 3.6 14V6z"/><path d="M3.6 6 10 9.6 16.4 6M10 9.6v7.8"/>',
    "data": '<ellipse cx="10" cy="5" rx="6" ry="2.4"/><path d="M4 5v10c0 1.3 2.7 2.4 6 2.4s6-1.1 6-2.4V5"/><path d="M4 10c0 1.3 2.7 2.4 6 2.4s6-1.1 6-2.4"/>',
    "chevron": '<path d="m6 8 4 4 4-4"/>',
    "clock": '<circle cx="10" cy="10" r="6.8"/><path d="M10 6.2V10l2.6 1.8"/>',
    "nb": '<rect x="3.5" y="2.5" width="13" height="15" rx="2"/><path d="M7 7h6M7 10h6M7 13h3.5"/>',
    "py": '<path d="M6 3.5h5.5L15 7v9.5H6z"/><path d="M11.5 3.5V7H15"/>',
}


def icon(name: str, cls: str = "pt-i") -> str:
    return (f'<svg class="{cls}" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">{_PATHS[name]}</svg>')


def anchor(key: str) -> str:
    """A dataset's entry on the Tutorial data page."""
    return "data-" + "".join(c if c.isalnum() else "-" for c in key.lower())


def page_data(env, docname: str) -> list[dict]:
    """The datasets a tutorial page reads, as the popup shows them. From tutorial_data (the registry of
    pytomography.datasets, with the datasets.fetch() line) when it is there, else from tutorials/datasets.yaml."""
    try:
        import tutorial_data
        if hasattr(env, "pytomo_data"):
            return tutorial_data.for_page(env, docname)
    except ImportError:
        pass
    t = getattr(env, "pytomo_by_notebook", {}).get(docname, {})
    known = getattr(env, "pytomo_datasets", {}) or {}
    users = {}
    for s in getattr(env, "pytomo_tutorials", []):
        for u in s["tutorials"]:
            for k in u.get("datasets") or []:
                users.setdefault(k, []).append({"notebook": u["notebook"], "title": u["title"]})
    out = []
    for key in t.get("datasets") or []:
        d = known.get(key)
        if not d:
            continue
        out.append({"key": key, "title": d.get("title", ""), "source": d.get("source", ""), "url": d.get("url", ""),
                    "licence": d.get("licence", ""), "cite": d.get("cite", ""), "download": d.get("size", ""), "disk": "",
                    "fetch": "", "steps": d.get("download", ""), "needs": "", "available": True, "note": "",
                    "anchor": anchor(key),
                    "used_by": [u for u in users.get(key, []) if u["notebook"] != t.get("notebook")]})
    return out


def run_text(run: dict | None) -> str:
    """When the notebook last ran end to end, as docs/tools/run_tutorials.py records it, in one short line."""
    if not run:
        return ""
    minutes = (run.get("wall_time_s") or 0) / 60
    memory = ", ".join(f"{run[k]:g} GB {label}" for k, label in (("peak_ram_gb", "RAM"), ("peak_gpu_gb", "GPU"))
                       if run.get(k))
    parts = [run.get("date") and f"Last run {run['date']}", run.get("gpu"),
             f"{minutes:.0f} min" if minutes >= 1 else "under a minute", memory and f"peak {memory}"]
    return " · ".join(p for p in parts if p)


def header_html(t: dict, *, colab: str, github: str, has_script: bool, run: dict | None, data: list[dict],
                data_page: str, viewer: dict | None) -> str:
    """The header under a tutorial's title.

    t          the tutorial's entry in tutorials.yaml (summary, modality, data, topic)
    run        the notebook's pytomography_run metadata, or None
    data       page_data(): the datasets it reads
    data_page  the Tutorial data page, relative to this page
    viewer     pytomo_viewer.for_page(): {"block", "thumb", "layers", "size"}, or None without 3D images
    """
    esc = html.escape
    tags = "".join(f'<span class="pt-tag">{esc(t[k])}</span>' for k in ("modality", "data", "topic")
                   if t.get(k) and t[k] not in ("None", "Any"))
    when = run_text(run)
    meta = f'<div class="pt-tut-meta"><span class="pt-tut-run">{icon("clock")}{esc(when)}</span></div>' if when else ""

    def column(label: str, body: str) -> str:
        return f'<div class="pt-tcol"><p class="pt-tcol-k">{label}</p><div class="pt-tcol-do">{body}</div></div>'

    cols = [column("Run it", f'<a class="pt-tb pt-tb-go" href="{esc(colab)}">{icon("play")}Open in Colab</a>'
                             f'<a class="pt-tb" href="{esc(github)}">{icon("code")}View on GitHub</a>')]
    def size(e: dict) -> str:   # "11.2 MB" on the button; a longer note ("included in SPECT.zip") only in the popup
        s = e.get("download", "") if e.get("available", True) else "not yet"
        return f"<small>{esc(s)}</small>" if s and len(s) <= 10 else ""

    if data:
        cols.append(column("Data", "".join(
            f'<button type="button" class="pt-dchip" data-key="{esc(e["key"])}" aria-haspopup="dialog" aria-expanded="false" '
            f'aria-controls="pt-dpop" title="What it is and where to get it">{icon("data")}<code>{esc(e["key"])}</code>'
            f'{size(e)}{icon("chevron", "pt-i pt-chev")}</button>'
            for e in data)))
    if viewer:
        # dark-light: the theme gives other images a white backdrop in dark mode, which showed as bars beside the picture
        thumb = (f'<img class="dark-light" src="{esc(viewer["thumb"])}" alt="" loading="lazy">'
                 if viewer.get("thumb") else icon("cube"))
        cols.append(column("Results",
                           f'<button type="button" class="pt-tb pt-tb-3d" data-ptv-open="{esc(viewer["block"])}" '
                           f'title="{esc(viewer.get("layers", ""))}">{thumb}<span>View results in 3D</span>'
                           f'<small>{esc(viewer.get("size", ""))}</small></button>'))
    panel = f'<div class="pt-tut-panel" data-cols="{len(cols)}" data-page="{esc(data_page)}">{"".join(cols)}</div>'
    tabs = ('<div class="pt-viewswitch" role="group" aria-label="Show this tutorial as">'
            f'<button type="button" data-view="notebook" aria-pressed="true">{icon("nb")}Jupyter notebook (.ipynb)</button>'
            f'<button type="button" data-view="script" aria-pressed="false">{icon("py")}Script (.py)</button></div>') if has_script else ""
    tabrow = f'<div class="pt-tut-tabs">{tabs}<span class="pt-tb-end"></span></div>'
    island = (f'<script type="application/json" class="pt-data">{json.dumps(data).replace("</", "<" + chr(92) + "/")}</script>'
              if data else "")
    # the tags follow the summary, on its line
    lede = (f'<p class="pt-tut-lede">{esc(t.get("summary", ""))} <span class="pt-tut-tags">{tags}</span></p>'
            if t.get("summary") or tags else "")
    return f'<div class="pt-tut pt-launch">{lede}{meta}{panel}{tabrow}{island}</div>'


def drop_data_note(section) -> None:
    """Remove the notebook's "Data." note (a quote that starts with **Data.**): the header's data popup says the same.
    The notebook keeps it for Jupyter and Colab readers until Data Chat takes it out."""
    for q in list(section.findall(nodes.block_quote)):
        para = next(iter(q.findall(nodes.paragraph)), None)
        # the first child that isn't empty text (MyST starts the paragraph with an empty one)
        first = next((c for c in (para.children if para is not None else []) if not isinstance(c, nodes.Text) or c.strip()), None)
        if isinstance(first, nodes.strong) and first.astext().strip() == "Data.":
            q.parent.remove(q)
            return


def data_page_href(docname: str) -> str:
    return posixpath.relpath("tutorials/data", posixpath.dirname(docname)) + ".html"
