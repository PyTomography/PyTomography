"""The 3D image viewer on tutorial pages and gallery cards.

Every tutorial listed in ``tutorials/viewer_images.json`` gets a "View the images in 3D" block under its launch bar,
and its gallery card a "View in 3D" button. A page loads only a small loader (``_static/viewer/pt-viewer-loader.js``);
the viewer itself (``pt-viewer.js``, about 60 kB) and the images (a few MB, from the image host) load when a reader
opens it.

``tutorials/viewer_images.json`` is written by ``docs/tools/viewer_upload.py`` when a release's images are uploaded::

    {"host": "https://huggingface.co/datasets/PyTomography/tutorial-images/resolve/{revision}/{path}",
     "revision": "<the host's commit>", "prefix": "v4.0.0",
     "tutorials": {"t_dicomdata": {"title": ..., "layers": [...], "bytes": ..., "manifest_sha256": ...}}}

The host is that one URL template, so moving the images (to GitHub Pages, say) changes one line; the
``pytomo_viewer_host`` setting in conf.py overrides it. Without the file, no page gets a viewer.

To preview the images of a local tutorial run instead, set ``PYTOMOGRAPHY_VIEWER_IMAGES`` to the run's viewer folder
(``docs/build/tutorial_runs/<run>/viewer``); the build then copies it into the site as ``_viewer/``.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import posixpath
import shutil
from pathlib import Path

from docutils import nodes

INDEX = "tutorials/viewer_images.json"
LOCAL_ENV = "PYTOMOGRAPHY_VIEWER_IMAGES"
LOCAL_DIR = "_viewer"


def _summary(name: str, man: dict) -> dict:
    return {"title": man.get("title", name), "description": man.get("description", ""),
            "layers": [L.get("label", L["name"]) for L in man.get("layers", [])], "bytes": man.get("bytes", 0)}


def _load(app) -> dict:
    """{notebook: {manifest, thumb, absolute, title, layers, bytes}} for every tutorial with images."""
    local = os.environ.get(LOCAL_ENV)
    found = {}
    if local:
        for path in sorted(Path(local).glob("*/manifest.json")):
            name = path.parent.name
            found[name] = dict(_summary(name, json.loads(path.read_text(encoding="utf8"))), absolute=False,
                               manifest=f"{LOCAL_DIR}/{name}/manifest.json", thumb=f"{LOCAL_DIR}/{name}/thumb.png")
        return found
    path = Path(app.srcdir) / INDEX
    if not path.exists():
        return found
    index = json.loads(path.read_text(encoding="utf8"))
    host = app.config.pytomo_viewer_host or index["host"]
    for name, t in index.get("tutorials", {}).items():
        url = lambda f: host.format(revision=index["revision"], path=f"{index['prefix']}/{name}/{f}")
        found[name] = dict(t, absolute=True, manifest=url("manifest.json"), thumb=url("thumb.png") if t.get("thumb", True) else "")
    return found


def builder_inited(app):
    app.env.pytomo_viewer = _load(app)
    # the loader fetches the viewer on demand; this stamp in its URLs keeps browsers from using an old copy
    h = hashlib.sha1()
    for f in ("pt-viewer.js", "pt-viewer.css"):
        path = Path(app.srcdir) / "_static" / "viewer" / f
        h.update(path.read_bytes() if path.exists() else b"")
    app.pytomo_viewer_version = h.hexdigest()[:8]


def _href(t: dict, key: str, here: str) -> str:
    return t[key] if t["absolute"] or not t[key] else posixpath.relpath(t[key], here or ".")


def _layers_text(t: dict) -> str:
    names = t["layers"]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"{listed} · {t['bytes'] / 1e6:.1f} MB" if t.get("bytes") else listed


def add_viewer(app, doctree):
    """A viewer block under the launch bar of each tutorial with images, and the image list on the gallery page."""
    env, docname = app.env, app.env.docname
    found = getattr(env, "pytomo_viewer", {})
    if not found:
        return
    env.note_dependency(str(Path(env.srcdir) / INDEX))
    here = posixpath.dirname(docname)
    section = next(iter(doctree.findall(nodes.section)), None)
    if env.doc2path(docname).suffix == ".ipynb" and Path(docname).name in found and section is not None:
        t = found[Path(docname).name]
        poster = (f'<img class="ptv-poster" src="{html.escape(_href(t, "thumb", here))}" alt="" loading="lazy">'
                  if t.get("thumb") else "")
        block = (f'<div class="ptv-block" data-manifest="{html.escape(_href(t, "manifest", here))}" data-title="{html.escape(t["title"])}">'
                 f'<button type="button" class="ptv-open">{poster}<span class="ptv-otext"><b>View the images in 3D</b>'
                 f'<small>{html.escape(_layers_text(t))}</small></span></button><p class="ptv-msg" role="status"></p></div>')
        # under the launch bar that pytomo_docs puts below the title
        at = next((i for i, n in enumerate(section.children) if isinstance(n, nodes.raw) and 'class="pt-launch"' in n.astext()),
                  next((i for i, n in enumerate(section.children) if isinstance(n, nodes.title)), -1))
        section.insert(at + 1, nodes.raw("", block, format="html"))
    if any(isinstance(n, nodes.raw) and "data-gallery" in n.astext() for n in doctree.findall(nodes.raw)):
        cards = {name: {"manifest": _href(t, "manifest", here), "title": t["title"]} for name, t in found.items()}
        island = json.dumps(cards).replace("<", "\\u003c")
        doctree += nodes.raw("", f'<script type="application/json" id="ptv-index">{island}</script>', format="html")


def add_assets(app, pagename, templatename, context, doctree):
    """Only pages with a viewer block or the gallery load the loader and its few lines of CSS."""
    body = context.get("body", "")
    if 'class="ptv-block"' in body or 'id="ptv-index"' in body:
        app.add_css_file("viewer/pt-viewer-page.css")
        app.add_js_file("viewer/pt-viewer-loader.js", loading_method="defer",
                        **{"data-version": getattr(app, "pytomo_viewer_version", "")})


def copy_local_images(app, exception):
    local = os.environ.get(LOCAL_ENV)
    if exception or not local or app.builder.format != "html":
        return
    target = Path(app.outdir) / LOCAL_DIR
    for path in Path(local).glob("*/manifest.json"):
        dest = target / path.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in [path, path.parent / "thumb.png", *path.parent.glob("*.nii.gz")]:
            if f.exists():
                shutil.copy2(f, dest / f.name)


def setup(app):
    app.add_config_value("pytomo_viewer_host", "", "env")
    app.connect("builder-inited", builder_inited)
    app.connect("doctree-read", add_viewer, priority=600)  # after pytomo_docs adds the launch bar
    app.connect("html-page-context", add_assets)
    app.connect("build-finished", copy_local_images)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
