"""The "Memory and speed" page (memory.md): tables generated at build time from the memory model's data, so that they
read without JavaScript and appear in the page's Markdown copy for AI agents.

* ``.. memory-reference::`` the tutorials' reconstructions (``docs/tools/memory_reference_cases.json``): their size,
  settings, the model's estimate (``docs/tools/memory_model.py`` evaluating ``_static/memory/model.json``) and what
  they measured.
* ``.. memory-scanners::`` PyTomography's sinogram of every scanner in ``_static/memory/pet_scanners.json``.
* ``.. memory-formulas::`` every model of ``model.json``: its derived quantities, its parts and its settings (the
  appendix of the formulas, which ``_static/memory/formulas.md`` writes out and the page includes).
* ``.. memory-sources::`` where the scanner and GPU data come from.

Each directive builds Markdown and parses it with MyST, so the page and its ``_md/memory.md`` copy (and
``llms-full.txt``) carry the same text: after ``pytomo_docs`` writes those copies, ``write_markdown`` here replaces each
directive's fence in them with its Markdown, and each ``{include}`` with the file it includes. The calculator reads the
reference cases at ``_static/memory/reference_cases.json``, copied from ``docs/tools`` at the end of the build.
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


# ---------------------------------------------------------------------------------------------- reference table
def _size_text(model_id: str, p: dict) -> str:
    image = f"{p.get('Lx')}×{p.get('Ly')}×{p.get('Lz')}"
    if model_id.startswith("pet") or model_id == "sss":
        text = f"{image}; {p.get('crystals_per_ring')} crystals × {p.get('rings')} rings"
        if p.get("tof"):
            text += f", TOF ({p['tof_bins']} bins)" if p.get("tof_bins") else ", TOF"
        if p.get("events"):
            text += f"; {_count(p['events'])} events"
        return text
    if model_id.startswith("ct"):
        return f"{image}; {p.get('views')} views of {p.get('det_cols')}×{p.get('det_rows')}"
    if model_id == "starguide":
        return f"{image}; {p.get('sg_views')} views at {p.get('sg_angles')} angles"
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


def reference_markdown(d: dict) -> str:
    ref, model = d["ref"], d["model"]
    m = ref["machine"]
    lines = [
        f"Estimated with each tutorial's own settings for the machine they ran on ({m['os'].title()}, {m['gpu']}, "
        f"{m['ram_gb']} GB of RAM). *Measured* is the whole notebook's peak (RAM: committed memory; GPU: "
        "`torch.cuda.max_memory_allocated`), which includes everything else the notebook keeps, so it can be above or "
        "below one step's estimate. *Cell* is the reconstruction cell alone, where it was measured. Pick any of these "
        "in the calculator's *Start from a tutorial* list.",
        "",
        "| Tutorial | Step | Size | Settings | Estimate RAM / GPU (GB) | Measured RAM / GPU (GB) | Cell RAM / GPU (GB) |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {x["id"]: x["title"] for x in model.models()}
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
        cell_text = f"{_gb(cell['ram_peak_gb'] - cell['ram_before_gb'])} more / {_gb(cell['gpu_peak_torch_gb'])}" if cell else ""
        lines.append(
            f"| `{c['tutorial']}` | {_cell(c['step'])} ({_cell(names[c['model']].split(' (')[0])}) | {_cell(_size_text(c['model'], params))} "
            f"| {_cell(_settings_text(e['settings']))} | {_gb(e['ram_gb'])} / {_gb(e['gpu_gb'])} | {_gb(meas['ram_gb'])} / {_gb(meas['gpu_gb'])} | {cell_text} |")
    if missing:
        lines += ["", "Not modelled yet (the measured peak of the whole notebook, RAM / GPU):", ""]
        for c in missing:
            meas = c["measured"]["tutorial"]
            lines.append(f"- `{c['tutorial']}`, {c['step']}: {_gb(meas['ram_gb'])} / {_gb(meas['gpu_gb'])} GB. {c.get('notes', '')}".rstrip())
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- scanners
TYPE_NAMES = {"PET/CT": "PET/CT", "PET/MR": "PET/MR", "total-body": "long axial FOV", "brain": "brain", "preclinical": "preclinical"}


def scanners_markdown(d: dict) -> str:
    lines = [
        "A PyTomography sinogram keeps every pair of rings (span 1): $C/2$ angles, $C+1$ radial bins and $R^2$ planes "
        "for $C$ crystals per ring and $R$ rings, as float32, times the TOF bins. These are the sizes of one such "
        "sinogram (the data, the sensitivity sinogram and a dense additive term are one each); a reconstruction holds "
        "one subset of it at a time on the GPU.",
        "",
        "| Scanner | Type | Crystals × rings | Sinogram (angles × radial × planes) | GB without TOF | TOF |",
        "|---|---|---|---|---|---|",
    ]
    for s in d["scanners"]:
        C, R = s.get("crystals_per_ring"), s.get("rings")
        tof = f"yes, {s['tof_resolution_ps']:g} ps" if s.get("tof") and s.get("tof_resolution_ps") else ("yes" if s.get("tof") else "no")
        vb = (s.get("vendor_sinogram") or {}).get("tof_bins")
        if vb:
            tof += f" ({vb} bins: {_gb(4 * (C // 2) * (C + 1) * R * R * vb / 1e9)} GB)" if C and R else ""
        if not C or not R:
            lines.append(f"| {_cell(s['name'])} | {TYPE_NAMES.get(s['type'], s['type'])} | not published | | | {tof} |")
            continue
        size = 4 * (C // 2) * (C + 1) * R * R / 1e9
        lines.append(f"| {_cell(s['name'])} | {TYPE_NAMES.get(s['type'], s['type'])} | {C} × {R} | {C // 2} × {C + 1} × {R * R:,} | {_gb(size)} | {tof} |")
    lines += ["", "For the long axial field of view scanners a sinogram of every ring pair is too large to hold: "
              "reconstruct them in list mode (`PETLMSystemMatrix`)."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- formulas
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


def formulas_markdown(d: dict, fmt: str = "html") -> str:
    mj = d["model_json"]
    lines = [
        "Each model exactly as the calculator evaluates it, generated from the memory model file "
        "(<a href=\"_static/memory/model.json\">model.json</a>) when the site is built: its derived quantities, its parts and its "
        "settings, in the model's expression language (`c ? a : b` is a if c, else b; `**` is a power). "
        "Sizes are in bytes. The overheads and the way a plan searches are as described above.",
        "",
        "Derived quantities every model uses:",
        "",
        "| Name | Expression |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {_code(v)} |" for k, v in _derived_rows(mj.get("derived"))]
    lines.append("")
    for m in mj["models"]:
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
        if fmt == "html":
            lines += [f":::{{dropdown}} {m['title']}", ""] + body + ["", ":::", ""]
        else:
            lines += [f"#### {m['title']}", ""] + body + [""]
    return "\n".join(lines)


# -------------------------------------------------------------------------------------------------- sources
def sources_markdown(d: dict) -> str:
    lines = [
        "**PET scanners** (`pet_scanners.json`, compiled 11 October 2026). Each value comes from a NEMA performance "
        "paper, a vendor page or STIR's scanner table; values computed from published numbers are marked in the file. "
        "The sources of each scanner:",
        "",
    ]
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
            by_source.setdefault(g["source"], 0)
            by_source[g["source"]] += 1
    stand_ins = sum(1 for g in d["gpus"] if "or similar" in g["match"])
    lines += ["", f"**GPUs** (`gpus.json`, {len(d['gpus']) - stand_ins} GPUs and {stand_ins} stand-ins that Firefox reports "
              "in place of the real GPU, compiled 11 October 2026): names, memory sizes and the renderer-string fragments "
              "the calculator matches, from the vendors' specification pages. `docs/tools/build_gpus.py` writes the file "
              "from its hand-checked table (each row with its source) and checks that every known renderer string finds "
              "its own entry; edit the table and run `python docs/tools/build_gpus.py` to update it. Sources:", ""]
    for url, n in sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- [{urlparse(url).netloc.replace('www.', '')}{urlparse(url).path}]({url}) ({n} GPU{'s' if n > 1 else ''})")
    lines += ["", "**Memory measurements**: the tutorial runs of 10 and 11 October 2026 (Windows 11, NVIDIA GeForce RTX 5090, "
              "64 GB of RAM), recorded in `docs/tools/memory_reference_cases.json`."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ directives
GENERATORS = {
    "memory-reference": lambda d, fmt: reference_markdown(d),
    "memory-scanners": lambda d, fmt: scanners_markdown(d),
    "memory-formulas": formulas_markdown,
    "memory-sources": lambda d, fmt: sources_markdown(d),
}


_CACHE: dict = {}


def _data(env) -> dict:
    """The data, loaded once per build (kept out of the pickled environment)."""
    srcdir = Path(env.srcdir)
    files = [srcdir / STATIC / n for n in ("model.json", "pet_scanners.json", "gpus.json")] + [_tools(srcdir) / "memory_reference_cases.json"]
    key = (str(srcdir), tuple(f.stat().st_mtime for f in files))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _load(srcdir)
    return _CACHE[key]


class MemoryDirective(SphinxDirective):
    has_content = False

    def run(self):
        srcdir = Path(self.env.srcdir)
        for dep in (STATIC / "model.json", STATIC / "pet_scanners.json", STATIC / "gpus.json"):
            self.env.note_dependency(str(srcdir / dep))
        self.env.note_dependency(str(_tools(srcdir) / "memory_reference_cases.json"))
        self.env.note_dependency(__file__)                     # this generator: the page is rebuilt when it changes
        text = GENERATORS[self.name](_data(self.env), "html")
        container = nodes.container(classes=["pt-" + self.name])
        self.state.nested_parse(StringList(text.splitlines(), source=self.name), self.content_offset, container)
        return [container]


def _included(srcdir: Path, path: str, options: str) -> str:
    """The Markdown an ``{include}`` fence stands for: the file from ``:start-line:``, its headings ``:heading-offset:``
    levels down (not inside code fences)."""
    opts = dict(re.findall(r":([a-z-]+):\s*([^\n]*)", options))
    lines = (srcdir / path).read_text(encoding="utf8").splitlines()[int(opts.get("start-line", 0) or 0):]
    offset, fenced, out = int(opts.get("heading-offset", 0) or 0), False, []
    for line in lines:
        if line.startswith("```"):
            fenced = not fenced
        out.append("#" * offset + line if offset and not fenced and re.match(r"#+ ", line) else line)
    return "\n".join(out).strip()


def write_markdown(app, exception):
    """Replaces each directive's fence in the page's Markdown copy and llms-full.txt with its Markdown, and each
    ``{include}`` with the file it includes; copies the reference cases next to the calculator."""
    if exception or app.builder.format != "html":
        return
    srcdir, outdir = Path(app.srcdir), Path(app.outdir)
    (outdir / STATIC).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_tools(srcdir) / "memory_reference_cases.json", outdir / STATIC / "reference_cases.json")
    d = _data(app.env)
    fence = re.compile(r"```\{(memory-[a-z]+)\}\s*\n```")
    include = re.compile(r"```\{include\}\s+(_static/memory/\S+)\n((?::[a-z-]+:[^\n]*\n)*)```")
    for target in (outdir / "_md" / f"{PAGE}.md", outdir / "llms-full.txt"):
        if target.exists():
            text = target.read_text(encoding="utf8")
            text = fence.sub(lambda m: GENERATORS[m.group(1)](d, "md").rstrip(), text)
            text = include.sub(lambda m: _included(srcdir, m.group(1), m.group(2)), text)
            target.write_text(text, encoding="utf8")


def setup(app):
    for name in GENERATORS:
        app.add_directive(name, MemoryDirective)
    app.connect("build-finished", write_markdown, priority=150)  # after pytomo_docs writes the Markdown copies (100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
