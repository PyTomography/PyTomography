/* The Architecture page's "Data and conventions" tab (#io): every data source, the reader that converts it, the
   standard form and what uses it, drawn from io-conventions.json. Hovering a node traces its path; clicking it opens
   its details (conventions, steps, vendor rules, tutorials). The standard form and the vendor rules are tables that
   _ext/pytomo_io.py writes into the page, as is a text version of this map (hidden here, kept for agents).
   The map shows nothing but a loading bar until the data and the fonts are in. */
(function () {
  "use strict";
  const panel = document.getElementById("tab-io");
  const host = document.getElementById("pt-io");
  if (!panel || !host) return;
  const NARROW = 900;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const clean = (s) => String(s).replace(/\s*\(#\d+\)/g, "");      // the audit's bug numbers, not GitHub issues
  const el = (tag, cls, html) => { const x = document.createElement(tag); if (cls) x.className = cls; if (html != null) x.innerHTML = html; return x; };
  const stCls = (s) => "is-" + String(s).replace(".", "");
  const anchor = (key) => "data-" + key.toLowerCase().replace(/[^a-z0-9]/g, "-");

  // ---------------------------------------------------------------------------------------- the three views
  const views = [...panel.querySelectorAll(".pt-io-views [role=tab]")];
  function showView(id, push) {
    views.forEach((t) => {
      const on = t.getAttribute("aria-controls") === id;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (push) history.replaceState(null, "", "#" + id);
    if (id !== "io-map") closeDrawer();
    requestAnimationFrame(layout);
  }
  views.forEach((t, k) => {
    t.addEventListener("click", () => showView(t.getAttribute("aria-controls"), true));
    t.addEventListener("keydown", (ev) => {
      const d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      const n = views[(k + d + views.length) % views.length];
      n.focus();
      showView(n.getAttribute("aria-controls"), true);
    });
  });
  showView("io-map", false);

  // --------------------------------------------------------------------------------------------- loading
  host.classList.add("is-loading");
  host.innerHTML = `<div class="pt-io-loading" role="status"><div class="pt-io-progress" role="progressbar" aria-label="Loading the map"
    aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span></span></div><p>Loading the map</p></div>`;
  const bar = host.querySelector(".pt-io-progress");
  const progress = (f, label) => {
    const p = Math.round(100 * Math.max(0, Math.min(1, f)));
    bar.firstElementChild.style.width = p + "%";
    bar.setAttribute("aria-valuenow", String(p));
    if (label) host.querySelector(".pt-io-loading p").textContent = label;
  };

  async function load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const total = +r.headers.get("Content-Length") || 0;
    if (!r.body || !total) { progress(0.8); return r.json(); }
    const reader = r.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      progress(0.8 * Math.min(1, got / total));
    }
    const all = new Uint8Array(got);
    let at = 0;
    chunks.forEach((c) => { all.set(c, at); at += c.length; });
    return JSON.parse(new TextDecoder().decode(all));
  }

  // ------------------------------------------------------------------------------------------- the graph
  let D = null, N = {}, INTO = {}, built = false;
  const nodeEl = {}, stacks = [];
  let mapEl, grid, svg, seg, edgeEls = [];
  const state = { lane: "all", hover: null, open: null };

  function index(d) {
    D = d;
    BANDS = [{ id: "img", name: "Every modality" }, ...d.modalities.filter((m) => m.id !== "img")];
    ["sources", "readers", "standard", "used_by"].forEach((kind, col) => d[kind].forEach((n) => { N[n.id] = Object.assign({ kind, col }, n); INTO[n.id] = []; }));
    Object.values(N).forEach((n) => (n.to || []).forEach((t) => INTO[t].push(n.id)));
  }
  const walk = (id, next) => { const seen = new Set(); const go = (k) => next(k).forEach((x) => { if (!seen.has(x)) { seen.add(x); go(x); } }); go(id); return seen; };
  const down = (id) => walk(id, (k) => N[k].to || []);
  const up = (id) => walk(id, (k) => INTO[k]);
  const related = (id) => new Set([id, ...up(id), ...down(id)]);
  const label = (n) => n.short || n.name || n.title;
  const laneName = (lane) => (D.modalities.find((m) => m.id === lane) || { name: "Every reader" }).name;
  const bandOf = (n) => n.lane || "img";            // the shared nodes sit with the image files, at the top
  let BANDS = [];

  // ------------------------------------------------------------------------------------------- the map
  function nodeHTML(n) {
    if (n.kind === "sources") {
      const st = n.status === "verified" ? "" : `<span class="pt-io-st ${stCls(n.status)}">${esc(n.status)}</span>`;
      return `<b>${esc(n.short)}${st}</b><small>${esc(n.brief)}</small>`;
    }
    if (n.kind === "readers") return `<b><code>${esc(n.name)}</code></b><small>${esc(n.brief)}</small>`;
    return `<b>${esc(n.title)}</b><small>${esc(n.brief)}</small>`;
  }
  function node(n) {
    const b = el("button", "pt-io-node" + (n.status === "v4.1" || n.status === "dropped" ? " is-gone" : ""), nodeHTML(n));
    b.type = "button";
    b.dataset.id = n.id;
    b.setAttribute("aria-label", `${label(n)}${n.status && n.status !== "verified" ? " (" + n.status + ")" : ""}: ${n.brief}. Show details`);
    b.addEventListener("mouseenter", () => { state.hover = n.id; trace(); });
    b.addEventListener("mouseleave", () => { state.hover = null; trace(); });
    b.addEventListener("focus", () => { state.hover = n.id; trace(); });
    b.addEventListener("blur", () => { state.hover = null; trace(); });
    b.addEventListener("click", () => (state.open === n.id ? closeDrawer() : openDrawer(n.id, b)));
    nodeEl[n.id] = b;
    return b;
  }
  function stack(col, items, extra) {
    const s = el("div", "pt-io-stack");
    s.dataset.col = D.columns[col].title;
    if (extra) s.style.gridColumn = extra;
    items.forEach((n) => s.append(node(N[n.id])));
    stacks.push(s);
    return s;
  }

  function build() {
    mapEl = el("div", "pt-io-map is-hidden");
    const toolbar = el("div", "pt-io-bar");
    seg = el("div", "pt-io-seg");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Show one modality");
    [{ id: "all", name: "All" }, ...D.modalities.filter((m) => m.id !== "img")].forEach((m) => {
      const b = el("button", null, esc(m.name));
      b.type = "button";
      b.dataset.lane = m.id;
      b.setAttribute("aria-pressed", String(m.id === state.lane));
      b.addEventListener("click", () => setLane(m.id));
      seg.append(b);
    });
    const touch = window.matchMedia && matchMedia("(hover: none)").matches;
    toolbar.append(seg, el("p", "pt-io-hint", touch ? "Tap a name for its conventions, steps and tutorials." : "Hover to trace a path. Click for conventions, steps and tutorials."));
    grid = el("div", "pt-io-grid");
    svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "pt-io-svg");
    svg.setAttribute("aria-hidden", "true");
    grid.append(svg);
    D.columns.forEach((c) => grid.append(el("div", "pt-io-colhead", `<b>${esc(c.title)}</b><span>${esc(c.sub)}</span>`)));
    // bands: first what every modality shares (image files, their readers, object_meta), then one per modality
    BANDS.forEach((band) => {
      const lab = el("div", "pt-io-lane", esc(band.name));
      lab.dataset.lane = band.id;
      grid.append(lab);
      ["sources", "readers", "standard", "used_by"].forEach((kind, col) => {
        const mine = D[kind].filter((n) => bandOf(n) === band.id).sort((a, b) => (a.lane === null ? 0 : 1) - (b.lane === null ? 0 : 1));
        const s = stack(col, mine);
        s.dataset.lane = band.id;
        grid.append(s);
      });
    });
    mapEl.append(toolbar, grid);
    host.append(mapEl);
    built = true;
  }

  // one modality: its band, and what its sources reach in the shared band
  function visibleSet() {
    if (state.lane === "all") return null;
    const set = new Set();
    Object.values(N).forEach((n) => {
      if (n.lane === state.lane) { set.add(n.id); if (n.kind === "sources") down(n.id).forEach((x) => set.add(x)); }
    });
    D.standard.concat(D.used_by).filter((n) => n.lane === null).forEach((n) => set.add(n.id));
    return set;
  }
  function setLane(lane) {
    state.lane = lane;
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lane === lane)));
    const vis = visibleSet();
    Object.entries(nodeEl).forEach(([id, b]) => (b.hidden = !!vis && !vis.has(id)));
    stacks.forEach((s) => s.classList.toggle("is-empty", ![...s.children].some((c) => !c.hidden)));
    grid.querySelectorAll(".pt-io-lane").forEach((lab) => {
      const lane = lab.dataset.lane;
      const mine = stacks.filter((s) => s.dataset.lane === lane);
      const empty = mine.every((s) => s.classList.contains("is-empty"));
      lab.hidden = empty;
      mine.forEach((s) => (s.hidden = empty));
    });
    if (state.open && nodeEl[state.open].hidden) closeDrawer();
    layout();
  }

  // ------------------------------------------------------------------------------------------- the lines
  function layout() {
    if (!built || panel.hidden || document.getElementById("io-map").hidden) return;
    const narrow = host.clientWidth < NARROW;
    mapEl.classList.toggle("is-narrow", narrow);
    svg.textContent = "";
    edgeEls = [];
    if (!narrow) {
      const G = grid.getBoundingClientRect();
      const at = (id) => {
        const b = nodeEl[id];
        if (!b || b.hidden || !b.offsetParent) return null;
        const r = b.getBoundingClientRect(), t = b.firstElementChild.getBoundingClientRect();
        return { l: r.left - G.left, r: r.right - G.left, y: (t.top + t.bottom) / 2 - G.top };
      };
      Object.values(N).forEach((n) => (n.to || []).forEach((t) => {
        const A = at(n.id), B = at(t);
        if (!A || !B) return;
        const x1 = A.r + 3, x2 = B.l - 3, mx = (x1 + x2) / 2;
        const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
        p.setAttribute("d", `M${x1},${A.y} C${mx},${A.y} ${mx},${B.y} ${x2},${B.y}`);
        // lines into object_meta (every reader has one) and from one band to another show only on a traced path
        const later = t === "std-object-meta" || bandOf(n) !== bandOf(N[t]);
        if (later) p.setAttribute("class", "is-later");
        svg.append(p);
        edgeEls.push({ p, a: n.id, b: t, later });
      }));
    }
    trace();
  }
  function trace() {
    if (!built) return;
    const id = state.hover || state.open;
    const set = id ? related(id) : null;
    Object.entries(nodeEl).forEach(([k, b]) => {
      b.classList.toggle("is-dim", !!set && !set.has(k));
      b.classList.toggle("is-on", k === state.open);
    });
    edgeEls.forEach(({ p, a, b, later }) => {
      const on = !!set && set.has(a) && set.has(b);
      p.classList.toggle("is-on", on);
      p.classList.toggle("is-dim", !!set && !on);
      p.classList.toggle("is-later", later && !on);
    });
  }

  // ------------------------------------------------------------------------------------------- details
  const drawer = el("aside", "pt-io-drawer");
  drawer.setAttribute("role", "dialog");
  drawer.setAttribute("aria-modal", "false");
  drawer.setAttribute("aria-labelledby", "pt-io-dtitle");
  document.body.append(drawer);
  let opener = null;

  const link = (id) => `<button type="button" class="pt-io-link" data-go="${esc(id)}">${N[id].kind === "readers" ? `<code>${esc(N[id].name)}</code>` : esc(label(N[id]))}</button>`;
  const links = (ids) => ids.map(link).join(", ");
  const uniq = (xs) => [...new Set(xs)];
  const section = (title, body) => (body ? `<h3>${esc(title)}</h3>${body}` : "");
  const tutorials = (ids) => (ids.length ? `<p class="pt-io-links">${ids.map((t) => `<a href="notebooks/${esc(t)}.html">${esc(D.tutorials[t] || t)}</a>`).join("")}</p>` : "");
  const rulesHTML = (ids) => ids.map((r) => D.vendor_rules.find((x) => x.id === r)).filter(Boolean).map((r) =>
    `<p class="pt-io-rule"><span class="pt-io-st ${stCls(r.status)}">${esc(r.status)}</span><b>${esc(r.vendor)} · ${esc(r.data)}</b> ${esc(r.rule)}<small>${esc(clean(r.evidence))}</small></p>`).join("");
  const path = (rows) => `<div class="pt-io-path">${rows.filter((r) => r[1].length).map(([k, ids]) => `<div><span>${esc(k)}</span><p>${links(ids)}</p></div>`).join("")}</div>`;

  function detailHTML(id) {
    const n = N[id];
    let kicker, body = "";
    if (n.kind === "sources") {
      kicker = `${laneName(n.lane)} · source`;
      const readers = n.to, std = uniq(readers.flatMap((r) => N[r].to)), used = uniq(std.flatMap((s) => N[s].to));
      body += `<p>${esc(n.name)}. ${esc(n.formats.join(", "))}.</p>`;
      body += section("Read with", n.call ? `<pre><code>${esc(n.call)}</code></pre>` : `<p>${esc(D.statuses[n.status] || "")}.</p>`);
      body += section("Its conventions", `<dl>${D.convention_fields.filter((k) => n.conventions[k] && n.conventions[k] !== "-")
        .map((k) => `<dt>${esc(k[0].toUpperCase() + k.slice(1))}</dt><dd>${esc(n.conventions[k])}</dd>`).join("")}</dl>`);
      body += section("To the standard form", `<ol>${n.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>`);
      body += section("Path", readers.length ? path([["Reader", readers], ["Returns", std], ["Used by", used]]) : "");
      body += section("Checked", `<p><span class="pt-io-st ${stCls(n.status)}" style="margin:0 6px 0 0">${esc(n.status)}</span>${esc(n.checked === "-" ? D.statuses[n.status] : n.checked)}</p>`);
      body += section("Vendor rules", rulesHTML(n.rules));
      body += section("From v3", `<p>${esc(n.v3)}</p>`);
      body += section("Tutorials", tutorials(n.tutorials));
      body += section("Data", n.datasets.length ? `<p class="pt-io-links">${n.datasets.map((k) => `<a href="tutorials/data.html#${anchor(k)}"><code>${esc(k)}</code></a>`).join("")}</p>` : "");
    } else if (n.kind === "readers") {
      kicker = `${laneName(n.lane)} · reader`;
      const reads = INTO[id], used = uniq(n.to.flatMap((s) => N[s].to));
      body += section("Call", `<pre><code>${esc(n.call)}</code></pre><p>Returns ${esc(n.returns)}.</p>`);
      body += section("What it does", `<ol>${n.does.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>`);
      body += section("Path", path([["Reads", reads], ["Returns", n.to], ["Used by", used]]));
      body += section("Also", n.also.length ? `<ul>${n.also.map((s) => `<li><code>${esc(s)}</code></li>`).join("")}</ul>` : "");
      body += section("Replaces (v3)", `<p>${n.replaces.map((s) => `<code>${esc(s)}</code>`).join(", ")}</p>`);
      body += section("Tutorials", tutorials(uniq(reads.flatMap((s) => N[s].tutorials))));
    } else {
      kicker = n.kind === "standard" ? (n.lane ? `${laneName(n.lane)} · standard form` : "standard form · every reader") : "used by";
      body += `<p>${esc(n.brief[0].toUpperCase() + n.brief.slice(1))}.</p>`;
      body += section(n.kind === "standard" ? "Definition" : "What", `<ul>${n.lines.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>`);
      body += section("Path", path(n.kind === "standard" ? [["From", INTO[id]], ["Used by", n.to]] : [["From", INTO[id]]]));
      if (n.kind === "standard") body += `<p><button type="button" class="pt-io-link" data-view="io-standard">The standard form, as one table</button></p>`;
    }
    const title = n.kind === "readers" ? `<code>${esc(n.name)}</code>` : esc(label(n));
    return `<header><div><div class="k">${esc(kicker)}</div><h2 id="pt-io-dtitle">${title}</h2></div>` +
      `<button type="button" class="pt-io-x">Close</button></header><div class="pt-io-body">${body}</div>`;
  }
  // the sheet starts below the site's header, wherever the page is scrolled (the preview banner above it scrolls away)
  const header = document.querySelector(".bd-header");
  const place = () => { if (state.open && header && innerWidth > 720) drawer.style.top = Math.max(0, header.getBoundingClientRect().bottom) + "px"; else drawer.style.top = ""; };
  window.addEventListener("scroll", place, { passive: true });
  function openDrawer(id, from) {
    if (!N[id]) return;
    if (nodeEl[id].hidden) setLane("all");
    state.open = id;
    opener = from || nodeEl[id];
    drawer.innerHTML = detailHTML(id);
    drawer.classList.add("is-open");
    place();
    drawer.querySelector(".pt-io-x").addEventListener("click", closeDrawer);
    drawer.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => openDrawer(b.dataset.go, nodeEl[b.dataset.go])));
    drawer.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view, true)));
    drawer.querySelector(".pt-io-x").focus({ preventScroll: true });
    trace();
  }
  function closeDrawer() {
    if (!state.open) return;
    state.open = null;
    drawer.classList.remove("is-open");
    if (opener && document.body.contains(opener) && !panel.hidden) opener.focus({ preventScroll: true });
    opener = null;
    trace();
  }
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && state.open) closeDrawer(); });

  // ------------------------------------------------------------------------------------------- links in
  function fromHash() {
    const h = decodeURIComponent(location.hash.slice(1));
    if (!h.startsWith("io")) return;
    if (h === "io") return showView("io-map", false);
    if (["io-map", "io-standard", "io-vendors"].includes(h)) return showView(h, false);
    if (h.startsWith("io-rule-")) {
      showView("io-vendors", false);
      const row = document.getElementById(h);
      if (row) row.scrollIntoView({ block: "center" });
      return;
    }
    const id = h.slice(3);
    if (built && N[id]) { showView("io-map", false); openDrawer(id); }
  }
  window.addEventListener("hashchange", fromHash);

  // ------------------------------------------------------------------------------------------- start
  const fonts = () => (document.fonts && document.fonts.ready ? Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 2500))]) : Promise.resolve());
  load(host.dataset.src)
    .then((d) => { progress(0.85, "Drawing the map"); index(d); build(); setLane("all"); return fonts(); })
    .then(() => {
      progress(1);
      host.classList.remove("is-loading");
      host.querySelector(".pt-io-loading").remove();
      layout();
      requestAnimationFrame(() => mapEl.classList.remove("is-hidden"));
      fromHash();
      // on a phone the page's tab row scrolls: bring this tab into it
      const tab = document.getElementById("t-io");
      if (tab && !panel.hidden) tab.parentElement.scrollLeft = tab.offsetLeft - tab.parentElement.offsetLeft - 16;
    })
    .catch((err) => {
      host.classList.remove("is-loading");
      host.innerHTML = `<p class="pt-io-error">The map could not be loaded (${esc(err.message || err)}). Its content is below as text.</p>`;
      const text = panel.querySelector(".pt-io-static");
      if (text) text.classList.add("is-shown");
    });
  if (window.ResizeObserver) new ResizeObserver(() => requestAnimationFrame(layout)).observe(host);
  window.addEventListener("resize", () => { if (panel.hidden) closeDrawer(); requestAnimationFrame(layout); });
})();
