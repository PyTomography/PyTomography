"""The 3D image viewer on tutorial pages and gallery cards.

Every tutorial listed in ``tutorials/viewer_images.json`` gets "View results in 3D" in its header (tutorial_page.py,
through ``for_page``), and its gallery card a "View in 3D" button. The viewer opens full screen. Inside the tutorial,
a "View ... in 3D" button follows the cell that computes each image (found from the image's expression in
``tutorials/viewer.yaml``), and opens the viewer on that image. A page loads only a small loader
(``_static/viewer/pt-viewer-loader.js``);
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

import ast
import hashlib
import html
import json
import os
import posixpath
import re
import shutil
from pathlib import Path

import yaml
from docutils import nodes

INDEX = "tutorials/viewer_images.json"
SPECS = "tutorials/viewer.yaml"
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
    specs = Path(app.srcdir) / SPECS
    app.env.pytomo_viewer_specs = (yaml.safe_load(specs.read_text(encoding="utf8")) or {}).get("tutorials", {}) if specs.exists() else {}
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


_HELPERS = {"vx", "np", "numpy", "torch", "nib", "pydicom"}
_CUBE = ('<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" aria-hidden="true">'
         '<path d="M10 2.6 16.4 6v8L10 17.4 3.6 14V6z"/><path d="M3.6 6 10 9.6 16.4 6M10 9.6v7.8"/></svg>')


def _root_name(expr: str) -> str | None:
    """The notebook variable an image expression starts from (recon_OSEM[0].cpu() -> recon_OSEM); None for a helper
    call such as vx.ct_dicom(files_CT), which reads an input rather than a result."""
    try:
        node = ast.parse(str(expr), mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Tuple) and node.elts:
        node = node.elts[0]
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) and node.id not in _HELPERS else None


def _cells(doctree):
    """MyST-NB's code cells in page order, each with its source and whether it shows an image."""
    out = []
    for c in doctree.findall(nodes.container):
        if "cell" not in c.get("classes", []):
            continue
        src = next((n.astext() for n in c.findall(nodes.literal_block)), "")
        out.append((c, src, any(True for _ in c.findall(nodes.image))))
    return out


def _result_buttons(doctree, spec: dict, exported: list[str]):
    """After the cell that computes each image the tutorial exports (or after the plot that shows it, when the next
    cell draws it), a button that opens the viewer on that image. A last plot that shows several of them gets one
    that opens them for comparison."""
    results = []          # (variable, layer name, label)
    layers = spec.get("layers", [])
    # the reconstructions: SPECT and PET, or in a CT tutorial its CT and attenuation images (in SPECT and PET
    # tutorials those are inputs)
    colour = any(L.get("kind") in ("spect", "pet") for L in layers)
    for L in layers:
        label = L.get("label", L["name"])
        if L.get("kind") not in (("spect", "pet") if colour else ("ct", "mu")) or label not in exported:
            continue
        var = _root_name(L.get("array", ""))
        if var:
            results.append((var, L["name"], label if len(label) <= 30 else L["name"]))
    if not results:
        return
    cells = _cells(doctree)
    uses = lambda src, v: re.search(rf"\b{re.escape(v)}\b", src) is not None
    spots = {}            # cell index -> [(layer name, label)]
    for var, name, label in results:
        assign = re.compile(rf"^[ \t]*(?:[\w.]+[ \t]*,[ \t]*)*{re.escape(var)}[ \t]*(?:,[ \t]*[\w.]+[ \t]*)*=(?!=)", re.M)
        hits = [i for i, (_, src, _) in enumerate(cells) if assign.search(src)]
        if not hits:
            continue
        i = hits[-1]      # the value the viewer shows is the last one assigned
        if i + 1 < len(cells) and cells[i + 1][2] and uses(cells[i + 1][1], var):
            i += 1        # the next cell plots it: the button goes under the picture
        spots.setdefault(i, []).append((name, label))
    if len(results) > 1:  # the last picture that shows two or more of them
        many = [i for i, (_, src, img) in enumerate(cells) if img and sum(uses(src, v) for v, _, _ in results) > 1]
        if many and many[-1] not in spots:
            spots[many[-1]] = [(None, "them all")]
    for i, found in spots.items():
        cell = cells[i][0]
        name = found[0][0]
        what = " and ".join(lbl for _, lbl in found) if name else "them all"
        text = f"Compare {what} in 3D" if not name else f"View {what} in 3D"
        layer = f' data-ptv-layer="{html.escape(name)}"' if name else ""
        button = (f'<div class="ptv-result"><button type="button" class="ptv-result-btn" data-ptv-open="ptv-block"{layer}>'
                  f'{_CUBE}<span>{html.escape(text)}</span></button></div>')
        cell.parent.insert(cell.parent.index(cell) + 1, nodes.raw("", button, format="html"))


def for_page(env, docname: str) -> dict | None:
    """What the tutorial header's View results in 3D needs: the viewer block's id, the picture, what it shows and its
    size; None for a tutorial without 3D images."""
    t = getattr(env, "pytomo_viewer", {}).get(Path(docname).name)
    if not t or not docname.startswith("notebooks/"):
        return None
    here = posixpath.dirname(docname)
    return {"block": "ptv-block", "thumb": _href(t, "thumb", here) if t.get("thumb") else "", "layers": _layers_text(t).split(" · ")[0],
            "size": f"{t['bytes'] / 1e6:.1f} MB" if t.get("bytes") else ""}


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
        # the header's View results in 3D (tutorial_page) and the buttons after each result open the viewer from here
        block = (f'<div class="ptv-block" id="ptv-block" data-manifest="{html.escape(_href(t, "manifest", here))}" '
                 f'data-title="{html.escape(t["title"])}"><p class="ptv-msg" role="status"></p></div>')
        # under the header that pytomo_docs puts below the title
        at = next((i for i, n in enumerate(section.children) if isinstance(n, nodes.raw) and "pt-launch" in n.astext()),
                  next((i for i, n in enumerate(section.children) if isinstance(n, nodes.title)), -1))
        section.insert(at + 1, nodes.raw("", block, format="html"))
        spec = getattr(env, "pytomo_viewer_specs", {}).get(Path(docname).name)
        if spec:
            env.note_dependency(str(Path(env.srcdir) / SPECS))
            _result_buttons(doctree, spec, t["layers"])
    if any(isinstance(n, nodes.raw) and "data-gallery" in n.astext() for n in doctree.findall(nodes.raw)):
        cards = {name: {"manifest": _href(t, "manifest", here), "title": t["title"]} for name, t in found.items()}
        island = json.dumps(cards).replace("<", "\\u003c")
        doctree += nodes.raw("", f'<script type="application/json" id="ptv-index">{island}</script>', format="html")


def add_assets(app, pagename, templatename, context, doctree):
    """Only pages with a viewer block or the gallery load the loader and its few lines of CSS. Tutorial pages also get
    the full width, with their headings in the section navigation (css/pytomo-tutorial.css, js/pytomo-tutorial.js)."""
    if pagename.startswith("notebooks/"):
        app.add_css_file("css/pytomo-tutorial.css")
        app.add_js_file("js/pytomo-tutorial.js", loading_method="defer")
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
