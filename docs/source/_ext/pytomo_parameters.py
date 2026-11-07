"""The "SPECT parameters" page (spect_parameters.md): PyTomography's tables of collimators, detectors and cameras,
generated at build time from the package's own data (``src/pytomography/data``: ``collimators.csv``, ``detectors.csv``,
``cameras.json``), so the page always lists what ``io.SPECT.collimator``, ``detector`` and ``camera`` accept.

``.. spect-parameters::`` three tabs: collimators (with a filter), detectors, cameras. On the page they are a
sphinx-design ``tab-set``; in the page's Markdown copy (``_md/spect_parameters.md``) and ``llms-full.txt``,
``write_markdown`` replaces the directive's fence with the same tables, the tabs as headings.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

PAGE = "spect_parameters"
FILES = ("collimators.csv", "detectors.csv", "cameras.json")


def _data_dir(srcdir: Path) -> Path:
    return srcdir.parent.parent / "src" / "pytomography" / "data"


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def _mm(value: str) -> str:
    return "" if value in ("", None) else f"{float(value):g}"


def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _morphis(a: float, b: float, c: float, e: float) -> float:
    return a + b * math.sqrt(e + c * e * e)


def collimators_markdown(data: Path, fmt: str) -> str:
    rows = _rows(data / "collimators.csv")
    lede = ("SIMIND's codes and dimensions in mm, from its collimator database (version 9.0, and the codes only "
            "earlier versions list). Holes are measured across the flats; *rotated*: the hexagonal lattice turned "
            "90 degrees. Only parallel-hole collimators can be used.")
    if fmt == "md":
        lede = lede.replace("``", "`")
    body = [lede, ""]
    if fmt == "html":
        body += ['<input class="pt-par-filter" type="search" placeholder="Filter: code, camera or type (e.g. Symbia, HE, LEHR)" '
                 'aria-label="Filter the collimators">', ""]
    table = []
    for r in rows:
        shape = r["hole_shape"] + (", rotated" if r["rotated"] == "true" else "")
        kind = "" if r["kind"] == "parallel" else f" ({r['kind']})"
        table.append([f"`{r['code']}`", r["family"], (r["type"] or "") + kind, shape, _mm(r["hole_diameter"]),
                      _mm(r["septal_thickness"]), _mm(r["hole_length"]),
                      "9.0" if r["source"] == "SIMIND 9.0" else "earlier"])
    body.append(_table(["Code", "Camera", "Type", "Holes", "Hole (mm)", "Septa (mm)", "Length (mm)", "SIMIND"], table))
    return "\n".join(body)


def detectors_markdown(data: Path, fmt: str) -> str:
    rows = _rows(data / "detectors.csv")
    lede = "The crystal, with SIMIND 9.0's template cover and backscatter, in mm."
    if fmt == "md":
        lede = lede.replace("``", "`")
    table = [[f"`{r['label']}`", r["crystal"], _mm(r["crystal_thickness"]), f"{_mm(r['cover_thickness'])} {r['cover_material']}",
              f"{_mm(r['backscatter_thickness'])} {r['backscatter_material']}"] for r in rows]
    return "\n".join([lede, "", _table(["Label", "Crystal", "Crystal (mm)", "Cover (mm)", "Backscatter (mm)"], table)])


def cameras_markdown(data: Path, fmt: str) -> str:
    cams = json.loads((data / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    lede = ("The energy resolution model and intrinsic resolution. The only model so far is Morphis et al.'s, "
            "FWHM = a + b sqrt(E + c E^2), E and FWHM in keV.")
    if fmt == "md":
        lede = lede.replace("``", "`")
    table, sources = [], []
    for c in cams:
        er = c["energy_resolution"]
        a, b, k = er["a"], er["b"], er["c"]
        at = {e: _morphis(a, b, k, e) for e in (140.0, 208.0, 440.0)}
        table.append([c["name"], ", ".join(c.get("aliases", [])) or "", er["model"], f"a = {a:g} keV, b = {b:g}, c = {k:g} /keV",
                      f"{100 * at[140.0] / 140:.1f} %", f"{at[208.0]:.1f} keV", f"{at[440.0]:.1f} keV",
                      f"{c['intrinsic_fwhm_140keV']:g} mm"])
        sources += [f"* {c['name']}, energy resolution: {er['source']}", f"* {c['name']}, intrinsic resolution: {c['intrinsic_source']}"]
    out = [lede, "", _table(["Camera", "Also", "Energy resolution", "Parameters", "FWHM at 140 keV", "at 208 keV",
                             "at 440 keV", "Intrinsic FWHM (140 keV)"], table), "", "Sources:", "", *sources]
    return "\n".join(out)


TABS = (("Collimators", collimators_markdown), ("Detectors", detectors_markdown), ("Cameras", cameras_markdown))


def page_markdown(data: Path, fmt: str) -> str:
    if fmt == "html":
        out = ["::::::{tab-set}", ""]
        for label, make in TABS:
            out += [f":::::{{tab-item}} {label}", "", make(data, fmt), "", ":::::", ""]
        out.append("::::::")
        return "\n".join(out)
    return "\n\n".join(f"## {label}\n\n{make(data, fmt)}" for label, make in TABS)


class ParametersDirective(SphinxDirective):
    has_content = False

    def run(self):
        data = _data_dir(Path(self.env.srcdir))
        for name in FILES:
            self.env.note_dependency(str(data / name))
        self.env.note_dependency(__file__)
        container = nodes.container(classes=["pt-spect-parameters"])
        text = page_markdown(data, "html")
        self.state.nested_parse(StringList(text.splitlines(), source=self.name), self.content_offset, container)
        return [container]


def write_markdown(app, exception):
    """Replaces the directive's fence in the page's Markdown copy and llms-full.txt with the tables."""
    if exception or app.builder.format != "html":
        return
    outdir = Path(app.outdir)
    data = _data_dir(Path(app.srcdir))
    fence = re.compile(r"```\{spect-parameters\}\s*\n```")
    for target in (outdir / "_md" / f"{PAGE}.md", outdir / "llms-full.txt"):
        if target.exists():
            text = target.read_text(encoding="utf8")
            target.write_text(fence.sub(lambda m: page_markdown(data, "md"), text), encoding="utf8")


def setup(app):
    app.add_directive("spect-parameters", ParametersDirective)
    app.connect("build-finished", write_markdown, priority=150)  # after pytomo_docs writes the Markdown copies (100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
