"""The Architecture page's "Data and conventions" tab (architecture.md, ``#io``): every data source, its conventions,
the reader that converts it and the standard form, from ``_static/arch/io-conventions.json``.

* ``.. io-map::`` every source, reader, standard object and user of it as text: the content of the interactive map
  (``_static/arch/pt-io.js`` draws the map from the same JSON and hides this text).
* ``.. io-standard::`` the standard form, one table.
* ``.. io-vendors::`` the vendor rules, one table (the library's ``pytomography/io/_vendors.py``).

Each directive writes HTML for the page and Markdown for the page's Markdown copy (``_md/architecture.md``) and
``llms-full.txt``, where ``write_markdown`` replaces the directive's fence, as the memory page does.

The build checks the JSON against the library and the tutorials, and warns when they drift apart: its vendor rules
against ``src/pytomography/io/_vendors.py`` when the library has one (``docs/tools/io_vendor_rules.py --write`` copies
them), and every tutorial it names against ``tutorials/tutorials.yaml``.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

from docutils import nodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

PAGE = "architecture"
DATA = Path("_static") / "arch" / "io-conventions.json"
FIELDS = [("frame", "Frame"), ("axes", "Axes"), ("units", "Units"), ("origin", "Origin"), ("angles", "Angles"),
          ("values", "Values"), ("quirks", "Quirks")]
STATUS = {"verified": "verified", "check": "check", "unverified": "unverified", "v4.1": "v4.1", "dropped": "dropped"}
_AUDIT_REF = re.compile(r"\s*\(#\d+\)")      # the I/O audit's bug numbers in _vendors.py, not GitHub issues

e = lambda s: html.escape(str(s), quote=True)


def _load(srcdir: Path) -> dict:
    return json.loads((srcdir / DATA).read_text(encoding="utf8"))


def _clean(text: str) -> str:
    return _AUDIT_REF.sub("", str(text))


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _codes(text: str) -> str:
    """Markdown code span (backticks inside are rare: none in the data)."""
    return "`" + str(text).replace("`", "'") + "`"


def _tut_link(nb: str, d: dict, fmt: str, here: str = "") -> str:
    title = d["tutorials"].get(nb, nb)
    if fmt == "md":
        return f"[{title}](notebooks/{nb}.md)"
    return f'<a href="{here}notebooks/{e(nb)}.html">{e(title)}</a>'


def _graph(d: dict) -> dict:
    """Every node by id, and who points to each."""
    nodes_ = {}
    for kind in ("sources", "readers", "standard", "used_by"):
        for n in d[kind]:
            nodes_[n["id"]] = dict(n, kind=kind)
    into = {i: [] for i in nodes_}
    for n in nodes_.values():
        for t in n.get("to", []):
            into[t].append(n["id"])
    return {"nodes": nodes_, "into": into}


def _name(n: dict) -> str:
    return n.get("short") or n.get("name") or n.get("title")


def _lane(d: dict, lane) -> str:
    return next((m["name"] for m in d["modalities"] if m["id"] == lane), "Every reader")


# --------------------------------------------------------------------------------------------------- io-map
def map_markdown(d: dict) -> str:
    g, rules = _graph(d), {r["id"]: r for r in d["vendor_rules"]}
    out = ["Every data source PyTomography reads, its own conventions, and the reader that brings it to the standard "
           f"form. As built in {d['library']}. The same data as JSON: "
           "[`_static/arch/io-conventions.json`](../_static/arch/io-conventions.json).", ""]
    for m in d["modalities"]:
        out += [f"### {m['name']}", ""]
        for s in [s for s in d["sources"] if s["lane"] == m["id"]]:
            out += [f"#### {s['short']} ({s['status']})", "", f"{s['name']}. Formats: {', '.join(s['formats'])}.", ""]
            if s["call"]:
                out += ["```python", s["call"], "```", ""]
            out += ["| Its conventions | |", "|---|---|"]
            out += [f"| {label} | {_cell(s['conventions'][k])} |" for k, label in FIELDS if s["conventions"][k] not in ("-", "")]
            out += ["", "To the standard form:", ""] + [f"{i}. {x}" for i, x in enumerate(s["steps"], 1)] + [""]
            path = [g["nodes"][t] for t in s["to"]]
            if path:
                out.append("- Read by: " + ", ".join(_codes(_name(p)) for p in path) + ".")
            out.append(f"- Checked: {s['checked']}")
            out.append(f"- From v3: {s['v3']}")
            if s["rules"]:
                out.append("- Vendor rules: " + "; ".join(f"`{r}` ({rules[r]['status']})" for r in s["rules"]) + ".")
            if s["datasets"]:
                out.append("- Datasets: " + ", ".join(_codes(x) for x in s["datasets"]) + ".")
            if s["tutorials"]:
                out.append("- Tutorials: " + ", ".join(_tut_link(t, d, "md") for t in s["tutorials"]) + ".")
            out.append("")
    out += ["### Readers", ""]
    for r in d["readers"]:
        out += [f"#### {r['name']}", "", "```python", r["call"], "```", "", f"Returns {r['returns']}.", ""]
        out += [f"{i}. {x}" for i, x in enumerate(r["does"], 1)] + [""]
        reads = [g["nodes"][i] for i in g["into"][r["id"]]]
        if reads:
            out.append("- Reads: " + ", ".join(_name(x) for x in reads) + ".")
        out.append("- Returns to: " + ", ".join(g["nodes"][t]["title"] for t in r["to"]) + ".")
        if r["also"]:
            out.append("- Also: " + "; ".join(_codes(x) for x in r["also"]) + ".")
        out += ["- Replaces (v3): " + ", ".join(_codes(x) for x in r["replaces"]) + ".", ""]
    out += ["### The standard form", ""]
    for s in d["standard"]:
        out += [f"#### {s['title']}", "", f"{s['brief'][0].upper()}{s['brief'][1:]}.", ""] + [f"- {x}" for x in s["lines"]]
        out += ["", "Used by: " + ", ".join(g["nodes"][t]["title"] for t in s["to"]) + ".", ""]
    out += ["### Used by", ""]
    for u in d["used_by"]:
        out += [f"#### {u['title']}", "", f"{u['brief'][0].upper()}{u['brief'][1:]}.", ""] + [f"- {x}" for x in u["lines"]] + [""]
    return "\n".join(out).rstrip() + "\n"


def map_html(d: dict) -> str:
    g, rules = _graph(d), {r["id"]: r for r in d["vendor_rules"]}
    out = ['<div class="pt-io-text">',
           f'<p>Every data source PyTomography reads, its own conventions, and the reader that brings it to the standard '
           f'form. As built in {e(d["library"])}. The same data as JSON: <a href="_static/arch/io-conventions.json">'
           f'<code>io-conventions.json</code></a>.</p>']
    for m in d["modalities"]:
        out.append(f'<h3>{e(m["name"])}</h3>')
        for s in [s for s in d["sources"] if s["lane"] == m["id"]]:
            out.append(f'<section class="pt-io-text-item" id="io-{e(s["id"])}"><h4>{e(s["short"])} '
                       f'<span class="pt-io-st is-{e(s["status"].replace(".", ""))}">{e(s["status"])}</span></h4>')
            out.append(f'<p>{e(s["name"])}. Formats: {e(", ".join(s["formats"]))}.</p>')
            if s["call"]:
                out.append(f'<pre><code>{e(s["call"])}</code></pre>')
            out.append('<dl>' + "".join(f'<dt>{label}</dt><dd>{e(s["conventions"][k])}</dd>' for k, label in FIELDS
                                        if s["conventions"][k] not in ("-", "")) + '</dl>')
            out.append('<p>To the standard form:</p><ol>' + "".join(f"<li>{e(x)}</li>" for x in s["steps"]) + '</ol>')
            if s["to"]:
                out.append('<p>Read by: ' + ", ".join(f'<code>{e(_name(g["nodes"][t]))}</code>' for t in s["to"]) + '.</p>')
            out.append(f'<p>Checked: {e(s["checked"])}</p><p>From v3: {e(s["v3"])}</p>')
            if s["rules"]:
                out.append('<p>Vendor rules: ' + "; ".join(f'<code>{e(r)}</code> ({e(rules[r]["status"])})' for r in s["rules"]) + '.</p>')
            if s["tutorials"]:
                out.append('<p>Tutorials: ' + ", ".join(_tut_link(t, d, "html") for t in s["tutorials"]) + '.</p>')
            out.append('</section>')
    out.append('<h3>Readers</h3>')
    for r in d["readers"]:
        out.append(f'<section class="pt-io-text-item" id="io-{e(r["id"])}"><h4><code>{e(r["name"])}</code></h4>'
                   f'<pre><code>{e(r["call"])}</code></pre><p>Returns {e(r["returns"])}.</p>'
                   '<ol>' + "".join(f"<li>{e(x)}</li>" for x in r["does"]) + '</ol>')
        reads = [g["nodes"][i] for i in g["into"][r["id"]]]
        if reads:
            out.append('<p>Reads: ' + e(", ".join(_name(x) for x in reads)) + '.</p>')
        out.append('<p>Returns to: ' + e(", ".join(g["nodes"][t]["title"] for t in r["to"])) + '.</p>')
        if r["also"]:
            out.append('<p>Also: ' + "; ".join(f"<code>{e(x)}</code>" for x in r["also"]) + '.</p>')
        out.append('<p>Replaces (v3): ' + ", ".join(f"<code>{e(x)}</code>" for x in r["replaces"]) + '.</p></section>')
    for title, kind in (("The standard form", "standard"), ("Used by", "used_by")):
        out.append(f'<h3>{title}</h3>')
        for s in d[kind]:
            out.append(f'<section class="pt-io-text-item" id="io-{e(s["id"])}"><h4>{e(s["title"])}</h4>'
                       f'<p>{e(s["brief"][0].upper() + s["brief"][1:])}.</p><ul>' + "".join(f"<li>{e(x)}</li>" for x in s["lines"]) + '</ul></section>')
    out.append('</div>')
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- io-standard
def standard_markdown(d: dict) -> str:
    out = ["| Topic | Convention | In code |", "|---|---|---|"]
    out += [f"| {r['topic']} | {_cell(r['rule'])} | {_cell(_codes(r['code']))} |" for r in d["convention"]]
    return "\n".join(out) + "\n"


def standard_html(d: dict) -> str:
    rows = "".join(f'<tr><th scope="row">{e(r["topic"])}</th><td class="c-rule">{e(r["rule"])}</td>'
                   f'<td class="c-code"><code>{e(r["code"])}</code></td></tr>' for r in d["convention"])
    return ('<table class="pt-io-table pt-io-standard"><thead><tr><th scope="col">Topic</th><th scope="col">Convention</th>'
            f'<th scope="col">In code</th></tr></thead><tbody>{rows}</tbody></table>')


# ----------------------------------------------------------------------------------------------- io-vendors
def vendors_markdown(d: dict) -> str:
    out = ["Every rule PyTomography applies because a vendor or a program writes its data its own way "
           "(`pytomography/io/_vendors.py`). Rules marked check or unverified warn (`UnverifiedVendorRule`) when "
           "they are used.", "",
           "| Vendor | Data | Rule | Evidence | Status |", "|---|---|---|---|---|"]
    out += [f"| {_cell(r['vendor'])} | {_cell(r['data'])} | {_cell(r['rule'])} | {_cell(_clean(r['evidence']))} | {r['status']} |"
            for r in d["vendor_rules"]]
    return "\n".join(out) + "\n"


def vendors_html(d: dict) -> str:
    rows = "".join(
        f'<tr id="io-rule-{e(r["id"])}"><th scope="row"><b>{e(r["vendor"])}</b><span>{e(r["data"])}</span></th>'
        f'<td class="c-rule">{e(r["rule"])}<span class="c-ev">{e(_clean(r["evidence"]))}</span></td>'
        f'<td class="c-st"><span class="pt-io-st is-{e(r["status"])}">{e(r["status"])}</span></td></tr>'
        for r in d["vendor_rules"])
    return ('<p class="pt-io-note">Rules marked check or unverified warn (<code>UnverifiedVendorRule</code>) when a '
            'reader uses them. From <code>pytomography/io/_vendors.py</code>.</p>'
            '<table class="pt-io-table pt-io-vendors"><thead><tr><th scope="col">Vendor and data</th>'
            f'<th scope="col">Rule and evidence</th><th scope="col">Status</th></tr></thead><tbody>{rows}</tbody></table>')


GENERATORS = {
    "io-map": (map_html, map_markdown),
    "io-standard": (standard_html, standard_markdown),
    "io-vendors": (vendors_html, vendors_markdown),
}


# -------------------------------------------------------------------------------------------------- checks
def _check(app, d: dict) -> None:
    """Warn when the JSON has drifted from the library's vendor table or from the tutorials."""
    srcdir = Path(app.srcdir)
    tools = srcdir.parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import io_vendor_rules
    library = io_vendor_rules.library_rules(srcdir.parent.parent / "src" / "pytomography" / "io" / "_vendors.py")
    if library is not None:
        diff = io_vendor_rules.differences(d["vendor_rules"], library)
        if diff:
            logger.warning(f"{DATA.as_posix()}: the vendor rules differ from src/pytomography/io/_vendors.py "
                           f"({'; '.join(diff[:4])}{'...' if len(diff) > 4 else ''}): run "
                           "python docs/tools/io_vendor_rules.py --write")
    known = {t["notebook"]: t["title"] for s in getattr(app.env, "pytomo_tutorials", []) for t in s["tutorials"]}
    if known:
        named = {t for s in d["sources"] for t in s["tutorials"]} | set(d["tutorials"])
        missing = sorted(t for t in named if t not in known)
        if missing:
            logger.warning(f"{DATA.as_posix()} names tutorials that tutorials.yaml does not list: {', '.join(missing)}")
        stale = sorted(t for t, title in d["tutorials"].items() if t in known and known[t] != title)
        if stale:
            logger.warning(f"{DATA.as_posix()}: tutorial titles differ from tutorials.yaml: {', '.join(stale)}")


class IODirective(SphinxDirective):
    has_content = False

    def run(self):
        srcdir = Path(self.env.srcdir)
        self.env.note_dependency(str(srcdir / DATA))
        self.env.note_dependency(__file__)
        d = _load(srcdir)
        if self.name == "io-map":
            _check(self.env.app, d)
        text = GENERATORS[self.name][0](d)
        return [nodes.raw("", text, format="html")]


def write_markdown(app, exception):
    """Replaces each directive's fence in the page's Markdown copy and llms-full.txt with its Markdown."""
    if exception or app.builder.format != "html":
        return
    d = _load(Path(app.srcdir))
    fence = re.compile(r"```\{(io-[a-z]+)\}\s*\n```")
    for target in (Path(app.outdir) / "_md" / f"{PAGE}.md", Path(app.outdir) / "llms-full.txt"):
        if target.exists():
            text = target.read_text(encoding="utf8")
            text = fence.sub(lambda m: GENERATORS[m.group(1)][1](d).rstrip(), text)
            target.write_text(text, encoding="utf8")


def setup(app):
    for name in GENERATORS:
        app.add_directive(name, IODirective)
    app.connect("build-finished", write_markdown, priority=160)  # after pytomo_docs writes the Markdown copies (100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
