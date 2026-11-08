"""The "Memory and speed" page (memory.md): its reference, formulas and sources, generated at build time from the memory
model's data, so that they read without JavaScript and appear in the page's Markdown copy for AI agents.

* ``.. memory-reference::`` two tabs: the tutorials' reconstructions (``docs/tools/memory_reference_cases.json``) with
  the model's estimate (``docs/tools/memory_model.py`` evaluating ``_static/memory/model.json``) and what they measured;
  PyTomography's sinogram of every scanner in ``_static/memory/pet_scanners.json``.
* ``.. memory-formulas::`` one tab per topic or model: the text of ``_static/memory/formulas.md`` for it, and the
  model's own derived quantities, parts and settings from ``model.json`` (collapsed).
* ``.. memory-sources::`` where the scanner and GPU data come from (collapsed).

Each directive builds Markdown and parses it with MyST. On the page, tabs and collapsed parts are sphinx-design
``tab-set`` and ``dropdown``: all their content is in the HTML. In the page's Markdown copy (``_md/memory.md``) and
``llms-full.txt``, ``write_markdown`` replaces each directive's fence with the same Markdown, the tabs as headings.
The calculator reads the reference cases at ``_static/memory/reference_cases.json``, copied from ``docs/tools`` at the
end of the build.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

PAGE = "memory"
STATIC = Path("_static") / "memory"
FILES = ("model.json", "pet_scanners.json", "gpus.json", "formulas.md")


def _tools(srcdir: Path) -> Path:
    return srcdir.parent / "tools"


def _load(srcdir: Path) -> dict:
    tools = _tools(srcdir)
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import memory_model  # docs/tools/memory_model.py: the model in pure Python
    static = srcdir / STATIC
    return {
        "model": memory_model.MemoryModel.load(str(static / "model.json")),
        "model_json": json.loads((static / "model.json").read_text(encoding="utf8")),
        "ref": json.loads((tools / "memory_reference_cases.json").read_text(encoding="utf8")),
        "scanners": json.loads((static / "pet_scanners.json").read_text(encoding="utf8")),
        "gpus": json.loads((static / "gpus.json").read_text(encoding="utf8")),
        "formulas": (static / "formulas.md").read_text(encoding="utf8"),
    }


def _gb(x: float) -> str:
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:.1f}" if x >= 1 else f"{x:.2f}"


def _cell(text: str) -> str:
    """Text for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _code(expr) -> str:
    text = expr if isinstance(expr, str) else json.dumps(expr)
    return "`" + text.replace("|", "\\|") + "`"


def _count(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3g} billion"
    if n >= 1e6:
        return f"{n / 1e6:.3g} million"
    return f"{n:,.0f}"


def _tabs(tabs: list, fmt: str, level: str = "###") -> str:
    """Tabs [(label, Markdown)]: a sphinx-design tab set on the page, headings in the Markdown copy."""
    out = []
    if fmt == "html":
        out += ["::::::{tab-set}", ""]
        for label, body in tabs:
            out += [f":::::{{tab-item}} {label}", "", body.strip(), "", ":::::", ""]
        out.append("::::::")
    else:
        for label, body in tabs:
            out += [f"{level} {label}", "", body.strip(), ""]
    return "\n".join(out)


def _collapsed(title: str, body: str, fmt: str, level: str = "####") -> str:
    """A part shown on request: a sphinx-design dropdown on the page, a heading in the Markdown copy."""
    if fmt == "html":
        return "\n".join([f":::{{dropdown}} {title}", "", body.strip(), "", ":::"])
    return "\n".join([f"{level} {title}", "", body.strip()])


# ---------------------------------------------------------------------------------------------- reference
def _size_text(model_id: str, p: dict) -> str:
    image = f"{p.get('Lx')}×{p.get('Ly')}×{p.get('Lz')}"
    if model_id.startswith("pet") or model_id == "sss":
        text = f"{image}; {p.get('crystals_per_ring')}×{p.get('rings')}"
        if p.get("tof"):
            text += f", TOF {p['tof_bins']}" if p.get("tof_bins") else ", TOF"
        if p.get("events"):
            text += f"; {_count(p['events'])} events"
        return text
    if model_id.startswith("ct"):
        return f"{image}; {p.get('views')} × {p.get('det_cols')}×{p.get('det_rows')}"
    if model_id == "starguide":
        return f"{image}; {p.get('sg_views')} views"
    return f"{image}; {p.get('views')} views" + (f", {p['n_beds']} beds" if p.get("n_beds") else "")


def _settings_text(settings: dict) -> str:
    def value(k, v):
        if v is None:
            return "auto"
        if k == "gpu_budget":
            return f"{v / 1e9:g} GB"
        if k == "memory_budget_gb":
            return f"{v:g} GB"
        return ("yes" if v else "no") if isinstance(v, bool) else str(v)
    return ", ".join(f"{k} {value(k, v)}" for k, v in settings.items()) or "none"


def tutorials_markdown(d: dict) -> str:
    ref, model = d["ref"], d["model"]
    m = ref["machine"]
    lines = [
        f"Each step estimated with its tutorial's settings ({m['os'].title()}, {m['gpu']}, {m['ram_gb']} GB). "
        "Measured: the whole notebook's peak; cell: that cell alone, above what was held before it. GB, RAM / GPU.",
        "",
        "| Tutorial | Step | Size | Settings | Estimate | Measured | Cell |",
        "|---|---|---|---|---|---|---|",
    ]
    missing = []
    for c in ref["cases"]:
        meas = c["measured"]["tutorial"]
        if not c.get("model"):
            missing.append(c)
            continue
        params = {**ref.get("defaults", {}), **c["params"]}
        known = {p["name"] for p in next(x for x in model.models() if x["id"] == c["model"])["params"]}
        params = {k: v for k, v in params.items() if k in known}
        e = model.estimate(c["model"], params, c.get("settings", {}))
        cell = c["measured"].get("cell")
        cell_text = f"{_gb(cell['ram_peak_gb'] - cell['ram_before_gb'])} / {_gb(cell['gpu_peak_torch_gb'])}" if cell else ""
        lines.append(
            f"| `{c['tutorial']}` | {_cell(c['step'])} | {_cell(_size_text(c['model'], params))} "
            f"| {_cell(_settings_text(e['settings']))} | {_gb(e['ram_gb'])} / {_gb(e['gpu_gb'])} | {_gb(meas['ram_gb'])} / {_gb(meas['gpu_gb'])} | {cell_text} |")
    if missing:
        lines += ["", "Not modelled yet (measured peak, RAM / GPU): " + "; ".join(
            f"`{c['tutorial']}` {_gb(c['measured']['tutorial']['ram_gb'])} / {_gb(c['measured']['tutorial']['gpu_gb'])} GB" for c in missing) + "."]
    return "\n".join(lines) + "\n"


TYPE_NAMES = {"PET/CT": "PET/CT", "PET/MR": "PET/MR", "total-body": "long axial FOV", "brain": "brain", "preclinical": "preclinical"}


def scanners_markdown(d: dict) -> str:
    lines = [
        "A PyTomography sinogram keeps every ring pair: $C/2$ angles × $(C+1)$ radial bins × $R^2$ planes, float32, "
        "times the TOF bins. Reconstruct long axial FOV scanners in list mode.",
        "",
        "| Scanner | Type | Crystals × rings | Sinogram | GB (no TOF) | TOF |",
        "|---|---|---|---|---|---|",
    ]
    for s in d["scanners"]:
        C, R = s.get("crystals_per_ring"), s.get("rings")
        tof = f"{s['tof_resolution_ps']:g} ps" if s.get("tof") and s.get("tof_resolution_ps") else ("yes" if s.get("tof") else "no")
        vb = (s.get("vendor_sinogram") or {}).get("tof_bins")
        if vb and C and R:
            tof += f", {vb} bins: {_gb(4 * (C // 2) * (C + 1) * R * R * vb / 1e9)} GB"
        if not C or not R:
            lines.append(f"| {_cell(s['name'])} | {TYPE_NAMES.get(s['type'], s['type'])} | not published | | | {tof} |")
            continue
        size = 4 * (C // 2) * (C + 1) * R * R / 1e9
        lines.append(f"| {_cell(s['name'])} | {TYPE_NAMES.get(s['type'], s['type'])} | {C} × {R} | {C // 2} × {C + 1} × {R * R:,} | {_gb(size)} | {tof} |")
    return "\n".join(lines) + "\n"


def reference_markdown(d: dict, fmt: str = "html") -> str:
    return _tabs([("Tutorials", tutorials_markdown(d)), ("PET scanners", scanners_markdown(d))], fmt)


# ------------------------------------------------------------------------------------------------ formulas
# Tabs of the formulas: (label, the formulas.md sections they take, by the start of their title, the models whose
# model.json tables they show). Sections and models no tab names go to "Other".
FORMULA_TABS = [
    ("Estimate", ["How an estimate is built"], []),
    ("Plan", ["How a plan is chosen"], []),
    ("Algorithms", ["Reconstruction algorithms"], []),
    ("PET sinogram", ["PET sinogram"], ["pet_sinogram"]),
    ("PET list mode", ["PET list mode"], ["pet_listmode"]),
    ("Scatter", ["Single scatter simulation"], ["sss"]),
    ("CT helical", ["Helical CT OS-SART"], ["ct_helical_sart"]),
    ("CT helical FBP", ["Helical CT filtered back projection"], ["ct_helical_fbp"]),
    ("CT cone beam", ["Cone-beam CT"], ["ct_cone_sart", "ct_cone_fdk"]),
    ("SPECT", ["SPECT (SPECTSystemMatrix)", "SPECT filtered back projection"], ["spect", "spect_fbp"]),
    ("StarGuide", ["StarGuide"], ["starguide"]),
    ("KEM", ["Kernel EM"], ["kem"]),
    ("Multi-bed", ["Multi-bed"], ["spect_multibed"]),
    ("Other", [], []),
    ("Limits", ["What the formulas leave out"], []),
]


def _formula_sections(text: str) -> tuple:
    """formulas.md as (its text before the first section, [(section title, Markdown)])."""
    intro, sections, current, fenced = [], [], None, False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("# "):
            continue
        heading = None if fenced else re.match(r"## (.+)", line)
        if heading:
            current = [heading.group(1).strip(), []]
            sections.append(current)
            continue
        (current[1] if current else intro).append(line)
    return "\n".join(intro).strip(), [(t, "\n".join(b).strip()) for t, b in sections]


def _derived_rows(group) -> list:
    if not group:
        return []
    items = group.items() if isinstance(group, dict) else group
    return [(k, v) for k, v in items]


def _name(template: str) -> str:
    return re.sub(r"\{([^}]*)\}", r"\1", template)


def _part_rows(model_json: dict, parts: list) -> list:
    rows = []
    for p in parts:
        if "include" in p:
            rows += _part_rows(model_json, model_json["part_sets"][p["include"]])
            continue
        if "bytes" in p:
            on = p.get("on", 0)
            where = "GPU" if on in (1, True, "1", "true") else "RAM" if on in (0, False, "0", "false") else f"GPU if {_code(on)}, else RAM"
            size = _code(p["bytes"])
        else:
            where = "RAM and GPU"
            size = f"RAM {_code(p.get('ram', 0))}; GPU {_code(p.get('gpu', 0))}"
        name = _name(p["name"])
        if "repeat" in p:
            r = p["repeat"]
            name += f" (one per {r['var']} = {r['from']} … {r['to']})"
        rows.append((name, p.get("scope", "held"), _code(p["when"]) if "when" in p else "", size, where))
    return rows


def _candidates_text(s: dict) -> str:
    if "candidates" not in s:
        return ""
    out = []
    for item in s["candidates"]["items"]:
        if "value" in item:
            out.append(_code(item["value"]) + (f" if {_code(item['if'])}" if "if" in item else ""))
        else:
            q = item["seq"]
            out.append(f"{_code(q['value'])} for {q['var']} = {_code(q['from'])} … {_code(q['to'])}" + (f" if {_code(q['if'])}" if "if" in q else ""))
    text = "; ".join(out)
    if s["candidates"].get("sort") == "asc":
        text += " (ascending)"
    return text


def _model_tables(mj: dict, m: dict) -> str:
    """One model of model.json as tables: its derived quantities, parts and settings."""
    derived = []
    for name in m.get("use", []):
        derived += _derived_rows(mj["derived_sets"][name])
    derived += _derived_rows(m.get("derived"))
    body = [f"`{m['library']}`", ""]
    if m.get("allocator_fraction") is not None:
        body += [f"Allocator fraction on Windows: {m['allocator_fraction']:g}.", ""]
    if derived:
        body += ["| Derived | Expression |", "|---|---|"] + [f"| `{k}` | {_code(v)} |" for k, v in derived] + [""]
    body += ["| Part | Scope | When | Bytes | Where |", "|---|---|---|---|---|"]
    body += [f"| {_cell(n)} | {sc} | {w} | {sz} | {wh} |" for n, sc, w, sz, wh in _part_rows(mj, m["parts"])]
    if m.get("settings"):
        body += ["", "| Setting | Candidates, fastest first |", "|---|---|"]
        for s in m["settings"]:
            extra = f" Forced to {_code(s['force'])} when not null." if "force" in s else ""
            body.append(f"| {_cell(s.get('label', s['name']))} | {_candidates_text(s)}{extra} |")
    return "\n".join(body)


def formulas_markdown(d: dict, fmt: str = "html") -> str:
    mj = d["model_json"]
    intro, sections = _formula_sections(d["formulas"])
    models = {m["id"]: m for m in mj["models"]}
    used_sections, used_models, tabs = set(), set(), []
    plan = []
    for label, prefixes, ids in FORMULA_TABS:
        mine = [s for s in sections if s[0] not in used_sections and any(s[0].startswith(p) for p in prefixes)]
        used_sections.update(s[0] for s in mine)
        ms = [models[i] for i in ids if i in models]
        used_models.update(m["id"] for m in ms)
        plan.append([label, mine, ms])
    for entry in plan:                                         # what no tab names
        if entry[0] == "Other":
            entry[1] = [s for s in sections if s[0] not in used_sections]
            entry[2] = [m for m in mj["models"] if m["id"] not in used_models]
    level = "####" if fmt == "md" else ""
    for label, mine, ms in plan:
        parts = [intro] if label == FORMULA_TABS[0][0] and intro else []
        for title, body in mine:
            if fmt == "html":
                body = body.replace("```math", "```{math}")
            parts.append(f"**{title}**\n\n{body}" if len(mine) > 1 or label in ("Other",) else body)
        full = []
        if label == FORMULA_TABS[0][0]:
            full.append("Every model starts from these:\n\n| Derived | Expression |\n|---|---|\n"
                        + "\n".join(f"| `{k}` | {_code(v)} |" for k, v in _derived_rows(mj.get("derived"))))
        for m in ms:
            full.append(f"**{m['title']}**\n\n{_model_tables(mj, m)}")
        if full:
            parts.append(_collapsed("As the calculator evaluates it (model.json)",
                                    "In the model's expression language: `c ? a : b` is a if c, else b; `**` is a power. "
                                    "Sizes in bytes.\n\n" + "\n\n".join(full), fmt, level="####"))
        if parts:
            tabs.append((label, "\n\n".join(parts)))
    return _tabs(tabs, fmt)


# -------------------------------------------------------------------------------------------------- sources
def sources_markdown(d: dict, fmt: str = "html") -> str:
    lines = ["**PET scanners** (`pet_scanners.json`, 11 October 2026): NEMA performance papers, vendor pages and STIR's "
             "scanner table; derived values are marked in the file.", ""]
    for s in d["scanners"]:
        links = [u for u in s.get("sources", []) if urlparse(u).scheme in ("http", "https")]
        local = len(s.get("sources", [])) - len(links)
        refs = ", ".join(f"[{urlparse(u).netloc.replace('www.', '')}]({u})" for u in links)
        if local:
            refs += ("; " if refs else "") + "PyTomography's GATE macro"
        lines.append(f"- {s['name']}: {refs}")
    by_source = {}
    for g in d["gpus"]:
        if g.get("source"):
            by_source[g["source"]] = by_source.get(g["source"], 0) + 1
    stand_ins = sum(1 for g in d["gpus"] if "or similar" in g["match"])
    lines += ["", f"**GPUs** (`gpus.json`, {len(d['gpus']) - stand_ins} GPUs and {stand_ins} Firefox stand-ins, "
              "11 October 2026), from the vendors' pages. `python docs/tools/build_gpus.py` regenerates the file from "
              "its table and checks the matching.", ""]
    for url, n in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- [{urlparse(url).netloc.replace('www.', '')}{urlparse(url).path}]({url}) ({n})")
    lines += ["", "**Measurements**: the tutorial runs of 10 and 11 October 2026 (Windows 11, RTX 5090, 64 GB), in "
              "`docs/tools/memory_reference_cases.json`."]
    return _collapsed("Sources", "\n".join(lines), fmt, level="##")


# ------------------------------------------------------------------------------------------------ directives
GENERATORS = {
    "memory-reference": reference_markdown,
    "memory-formulas": formulas_markdown,
    "memory-sources": sources_markdown,
}

_CACHE: dict = {}


def _data(env) -> dict:
    """The data, loaded once per build (kept out of the pickled environment)."""
    srcdir = Path(env.srcdir)
    files = [srcdir / STATIC / n for n in FILES] + [_tools(srcdir) / "memory_reference_cases.json"]
    key = (str(srcdir), tuple(f.stat().st_mtime for f in files))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _load(srcdir)
    return _CACHE[key]


class MemoryDirective(SphinxDirective):
    has_content = False

    def run(self):
        srcdir = Path(self.env.srcdir)
        for name in FILES:
            self.env.note_dependency(str(srcdir / STATIC / name))
        self.env.note_dependency(str(_tools(srcdir) / "memory_reference_cases.json"))
        self.env.note_dependency(__file__)                     # this generator: the page is rebuilt when it changes
        text = GENERATORS[self.name](_data(self.env), "html")
        container = nodes.container(classes=["pt-" + self.name])
        self.state.nested_parse(StringList(text.splitlines(), source=self.name), self.content_offset, container)
        return [container]


def write_markdown(app, exception):
    """Replaces each directive's fence in the page's Markdown copy and llms-full.txt with its Markdown (tabs and
    collapsed parts as headings); copies the reference cases next to the calculator."""
    if exception or app.builder.format != "html":
        return
    srcdir, outdir = Path(app.srcdir), Path(app.outdir)
    (outdir / STATIC).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_tools(srcdir) / "memory_reference_cases.json", outdir / STATIC / "reference_cases.json")
    d = _data(app.env)
    fence = re.compile(r"```\{(memory-[a-z]+)\}\s*\n```")
    for target in (outdir / "_md" / f"{PAGE}.md", outdir / "llms-full.txt"):
        if target.exists():
            text = target.read_text(encoding="utf8")
            text = fence.sub(lambda m: GENERATORS[m.group(1)](d, "md").rstrip(), text)
            target.write_text(text, encoding="utf8")


def setup(app):
    for name in GENERATORS:
        app.add_directive(name, MemoryDirective)
    app.connect("build-finished", write_markdown, priority=150)  # after pytomo_docs writes the Markdown copies (100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
