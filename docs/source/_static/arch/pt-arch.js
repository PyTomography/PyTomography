/* The Architecture page's interactive map of PyTomography, drawn from window.PT_ARCH (pt-arch-data.js).
   Modality-specific stages are lanes (SPECT, PET, CT); from the likelihood on, one shared core serves every lane.
   Every card opens a drawer with its details; the walkthrough steps through one reconstruction's code. */
(function () {
  "use strict";
  const D = window.PT_ARCH;
  if (!D) return;
  const NS = "http://www.w3.org/2000/svg";
  const NARROW = 900;                      // below this width (of the map), one modality, top to bottom
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const svgEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const root = (document.querySelector('meta[name="pt-arch-root"]') || {}).content || "";
  const href = (h) => (/^(https?:|#)/.test(h) ? h : root + h);

  const state = { modality: "all", walk: null, open: null };
  const laneStages = D.stages.filter((s) => s.lanes);
  const coreStages = D.stages.filter((s) => !s.lanes);
  const cardKey = (stage, lane) => (lane ? stage + ":" + lane : stage);

  // ---------- map ----------
  const host = document.getElementById("pt-arch-map");
  if (!host) return;
  host.classList.add("pt-arch-map");
  const grid = el("div", "pt-arch-grid");
  const svg = svgEl("svg", { class: "pt-arch-svg", "aria-hidden": "true" });
  host.append(grid);
  grid.append(svg);
  const cards = {};
  const verbs = [];

  // where long names may break: before a capitalised word, after a dot, underscore or comma
  const wbrName = (t) => esc(t).replace(/([a-z0-9])([A-Z])/g, "$1<wbr>$2").replace(/([A-Z])([A-Z][a-z])/g, "$1<wbr>$2");
  const wbrPath = (t) => esc(t).replace(/\./g, ".<wbr>");
  const wbrCode = (html) => html.replace(/<code>([\s\S]*?)<\/code>/g, (m, x) => "<code>" + x.replace(/([_.,(])/g, "$1<wbr>") + "</code>");

  function cardHTML(c, stage) {
    let h = `<h3>${wbrName(c.title)}</h3><span class="k">${wbrPath(c.k || "")}</span>`;
    if (c.pipe) h += `<div class="pt-arch-pipe">${c.pipe.map((p, i) => (i ? "<i>→</i>" : "") + `<span class="${p.p ? "p" : ""}">${esc(p.t)}</span>`).join("")}</div>`;
    if (c.items) h += `<ul>${c.items.map((t) => `<li>${wbrCode(t)}</li>`).join("")}</ul>`;
    h += `<span class="more">Details</span>`;
    return h;
  }

  function build() {
    const cols = ["22px", ...D.stages.map(() => "minmax(0, 1fr)")];
    grid.style.gridTemplateColumns = cols.join(" ");
    grid.style.gridTemplateRows = `auto repeat(${D.modalities.length}, auto)`;
    D.stages.forEach((s, i) => {
      const h = el("div", "pt-arch-head", `<span class="pt-arch-step">${i + 1}</span><b>${esc(s.title)}</b><span class="sub">${wbrPath(s.sub)}</span>`);
      h.style.gridColumn = String(i + 2);
      h.style.gridRow = "1";
      grid.append(h);
    });
    D.modalities.forEach((m, r) => {
      const lab = el("div", "pt-arch-lane", esc(m.name));
      lab.style.gridColumn = "1";
      lab.style.gridRow = String(r + 2);
      lab.dataset.lane = m.id;
      grid.append(lab);
    });
    D.stages.forEach((s, i) => {
      const lanes = s.lanes ? D.modalities.map((m) => m.id) : [null];
      lanes.forEach((lane, r) => {
        const key = cardKey(s.id, lane);
        const c = D.cards[key];
        if (!c) return;
        const b = el("button", "pt-arch-card" + (lane ? "" : " is-shared"), cardHTML(c, s));
        b.type = "button";
        b.dataset.key = key;
        b.dataset.stage = `${i + 1} · ${s.title}`;
        if (lane) b.dataset.lane = lane;
        b.setAttribute("aria-label", `${s.title}${lane ? " for " + lane.toUpperCase() : ""}: ${c.title}. Show details`);
        b.style.gridColumn = String(i + 2);
        b.style.gridRow = lane ? String(r + 2) : `2 / span ${D.modalities.length}`;
        b.addEventListener("click", () => openDrawer(key, b));
        b.addEventListener("mouseenter", () => hover(key, true));
        b.addEventListener("mouseleave", () => hover(key, false));
        grid.append(b);
        cards[key] = b;
      });
    });
    D.transitions.forEach((t) => { const v = el("div", "pt-arch-verb", t.label); v.dataset.after = t.after; grid.append(v); verbs.push(v); });
    const core = el("div", "pt-arch-core"); core.id = "pt-arch-core"; grid.prepend(core);
    const coreLab = el("div", "pt-arch-core-label", esc(D.core.label)); coreLab.id = "pt-arch-core-label"; grid.append(coreLab);
    const bnd = el("div", "pt-arch-boundary"); bnd.id = "pt-arch-boundary"; grid.append(bnd);
    const tag = el("div", "pt-arch-boundary-tag", D.core.boundary); tag.id = "pt-arch-boundary-tag"; grid.append(tag);
  }

  // the arrows of the current layout: [from card key, to card key, lane or null]
  function links() {
    const out = [];
    const mods = narrow() || state.modality !== "all" ? [visibleLane()] : D.modalities.map((m) => m.id);
    mods.forEach((lane) => {
      for (let i = 0; i < laneStages.length - 1; i++) out.push([cardKey(laneStages[i].id, lane), cardKey(laneStages[i + 1].id, lane), lane]);
      out.push([cardKey(laneStages[laneStages.length - 1].id, lane), coreStages[0].id, lane]);
    });
    for (let i = 0; i < coreStages.length - 1; i++) out.push([coreStages[i].id, coreStages[i + 1].id, null]);
    return out.filter(([a, b]) => cards[a] && cards[b]);
  }

  const narrow = () => host.clientWidth < NARROW;
  const visibleLane = () => (state.modality === "all" ? D.modalities[0].id : state.modality);

  function layout() {
    const isNarrow = narrow();
    const single = isNarrow || state.modality !== "all";
    const nRows = single ? 1 : D.modalities.length;
    host.classList.toggle("is-narrow", isNarrow);
    grid.style.gridTemplateRows = `auto repeat(${nRows}, auto)`;
    grid.querySelectorAll(".pt-arch-lane").forEach((x) => {
      x.hidden = single && x.dataset.lane !== visibleLane();
      x.style.gridRow = single ? "2" : String(D.modalities.findIndex((m) => m.id === x.dataset.lane) + 2);
    });
    // which cards show, and in what order on narrow screens
    let order = 0;
    D.stages.forEach((s) => {
      const lanes = s.lanes ? D.modalities.map((m) => m.id) : [null];
      lanes.forEach((lane) => {
        const b = cards[cardKey(s.id, lane)];
        if (!b) return;
        const show = !single || !lane || lane === visibleLane();
        b.hidden = !show;
        b.style.order = isNarrow ? String(order++) : "";
        if (isNarrow) { b.style.gridColumn = "1"; b.style.gridRow = "auto"; }
        else {
          const i = D.stages.indexOf(s);
          const r = single ? 0 : lane ? D.modalities.findIndex((m) => m.id === lane) : -1;
          b.style.gridColumn = String(i + 2);
          b.style.gridRow = lane ? String(r + 2) : `2 / span ${nRows}`;
        }
      });
    });
    verbs.forEach((v) => (v.hidden = isNarrow));
    draw();
    applyFocus();
  }

  function draw() {
    svg.textContent = "";
    const g0 = grid.getBoundingClientRect();
    const box = (k) => { const r = cards[k].getBoundingClientRect(); return { l: r.left - g0.left, r: r.right - g0.left, t: r.top - g0.top, b: r.bottom - g0.top, cx: (r.left + r.right) / 2 - g0.left, cy: (r.top + r.bottom) / 2 - g0.top }; };
    const isNarrow = narrow();
    links().forEach(([a, b, lane]) => {
      const A = box(a), B = box(b);
      const g = svgEl("g", { "data-from": a, "data-to": b, "data-lane": lane || "" });
      let d, tip;
      if (isNarrow) {
        const x = A.cx, y1 = A.b + 2, y2 = B.t - 8;
        d = `M${x},${y1} L${x},${y2}`;
        tip = `${x - 5},${y2} ${x + 5},${y2} ${x},${y2 + 7}`;
      } else {
        const y = Math.max(B.t + 18, Math.min(B.b - 18, A.cy));
        const x1 = A.r + 2, x2 = B.l - 8;
        const y0 = A.cy;
        d = Math.abs(y - y0) < 1 ? `M${x1},${y0} L${x2},${y}` : `M${x1},${y0} C${(x1 + x2) / 2},${y0} ${(x1 + x2) / 2},${y} ${x2},${y}`;
        tip = `${x2},${y - 5} ${x2},${y + 5} ${x2 + 7},${y}`;
      }
      g.append(svgEl("path", { d }), svgEl("polygon", { points: tip }));
      svg.append(g);
    });
    if (isNarrow) { ["pt-arch-core", "pt-arch-boundary", "pt-arch-boundary-tag", "pt-arch-core-label"].forEach((id) => (document.getElementById(id).hidden = true)); return; }
    // each label sits just above the arrow it names: the top lane's, or the core's
    const heads = grid.querySelectorAll(".pt-arch-head");
    const topLane = state.modality === "all" ? D.modalities[0].id : state.modality;
    verbs.forEach((v) => {
      const i = D.stages.findIndex((s) => s.id === v.dataset.after);
      const s1 = D.stages[i], s2 = D.stages[i + 1];
      const a = cardKey(s1.id, s1.lanes ? topLane : null), b = cardKey(s2.id, s2.lanes ? topLane : null);
      if (!cards[a] || !cards[b]) return;
      const A = box(a), B = box(b);
      const y = s1.lanes ? A.cy : Math.max(B.t + 18, Math.min(B.b - 18, A.cy));
      v.style.left = ((A.r + B.l) / 2) + "px";
      v.style.top = (y - 13) + "px";
    });
    // the shared core and the boundary in front of it
    const first = cards[coreStages[0].id].getBoundingClientRect(), last = cards[coreStages[coreStages.length - 1].id].getBoundingClientRect();
    const h0 = heads[D.stages.indexOf(coreStages[0])].getBoundingClientRect();
    const gap = parseFloat(getComputedStyle(grid).columnGap) || 40;
    const core = document.getElementById("pt-arch-core");
    Object.assign(core.style, { left: (first.left - g0.left - gap * 0.32) + "px", top: (h0.top - g0.top - 30) + "px",
      width: (last.right - first.left + gap * 0.32 + 10) + "px", height: (last.bottom - h0.top + 40) + "px" });
    core.hidden = false;
    const lab = document.getElementById("pt-arch-core-label");
    Object.assign(lab.style, { left: (first.left - g0.left) + "px", top: (h0.top - g0.top - 22) + "px" });
    lab.hidden = false;
    const bx = first.left - g0.left - gap / 2;
    const bnd = document.getElementById("pt-arch-boundary");
    Object.assign(bnd.style, { left: bx + "px", top: (h0.top - g0.top - 30) + "px", height: (last.bottom - h0.top + 40) + "px" });
    bnd.hidden = false;
    const tag = document.getElementById("pt-arch-boundary-tag");
    Object.assign(tag.style, { left: bx + "px", top: (last.bottom - g0.top + 18) + "px" });
    tag.hidden = false;
    host.style.paddingBottom = "52px";
  }

  // ---------- highlighting ----------
  function pathOf(key) {
    // the cards a card's data flows through: its lane up to the core, then the core
    const [stage, lane] = key.split(":");
    if (lane) return new Set([...laneStages.map((s) => cardKey(s.id, lane)), ...coreStages.map((s) => s.id)]);
    const mods = state.modality === "all" ? D.modalities.map((m) => m.id) : [state.modality];
    return new Set([...mods.flatMap((l) => laneStages.map((s) => cardKey(s.id, l))), ...coreStages.map((s) => s.id)]);
  }

  function applyFocus(hoverKey) {
    const walkKey = state.walk ? currentStep().card : null;
    const lane = state.modality === "all" ? null : state.modality;
    Object.entries(cards).forEach(([k, b]) => {
      const bl = b.dataset.lane;
      b.classList.toggle("is-dim", !!(lane && bl && bl !== lane));
      b.classList.toggle("is-on", k === walkKey || k === state.open);
    });
    const onPath = hoverKey ? pathOf(hoverKey) : null;
    svg.querySelectorAll("g").forEach((g) => {
      const l = g.dataset.lane;
      const dim = (lane && l && l !== lane) || (onPath && !(onPath.has(g.dataset.from) && onPath.has(g.dataset.to)));
      g.classList.toggle("is-dim", !!dim);
      g.classList.toggle("is-on", !!(onPath && !dim));
    });
    grid.querySelectorAll(".pt-arch-lane").forEach((x) => (x.style.opacity = lane && x.dataset.lane !== lane ? 0.35 : 1));
  }
  const hover = (key, on) => applyFocus(on ? key : null);

  // ---------- modality switch ----------
  const seg = document.getElementById("pt-arch-modality");
  if (seg) {
    [{ id: "all", name: "All" }, ...D.modalities].forEach((m) => {
      const b = el("button", null, esc(m.name));
      b.type = "button";
      b.dataset.m = m.id;
      b.setAttribute("aria-pressed", String(m.id === state.modality));
      b.addEventListener("click", () => setModality(m.id));
      seg.append(b);
    });
  }
  function setModality(m) {
    state.modality = m;
    if (seg) seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === m || (m === "all" && b.dataset.m === "all"))));
    if (state.walk && m !== "all" && state.walk.m !== m) startWalk(m, 0);
    layout();
  }

  // ---------- drawer ----------
  const scrim = el("div", "pt-arch-scrim");
  const drawer = el("aside", "pt-arch-drawer");
  drawer.setAttribute("role", "dialog");
  drawer.setAttribute("aria-modal", "true");
  drawer.setAttribute("aria-labelledby", "pt-arch-dtitle");
  drawer.hidden = true;
  document.body.append(scrim, drawer);
  let opener = null;

  function detailHTML(key) {
    const c = D.cards[key], d = c.detail || {};
    let h = `<header><div><div class="k">${esc(c.k || "")}</div><h2 id="pt-arch-dtitle">${esc(d.title || c.title)}</h2></div>` +
      `<button type="button" class="pt-arch-x" aria-label="Close">Close</button></header><div class="pt-arch-body">`;
    if (d.summary) h += `<p>${d.summary}</p>`;
    if (d.knows || d.ignores) {
      h += `<div class="pt-arch-kv">`;
      if (d.knows) h += `<div class="yes"><b>Knows</b><ul>${d.knows.map((x) => `<li>${x}</li>`).join("")}</ul></div>`;
      if (d.ignores) h += `<div class="no"><b>Never sees</b><ul>${d.ignores.map((x) => `<li>${x}</li>`).join("")}</ul></div>`;
      h += `</div>`;
    }
    if (d.formats) h += `<h4>Formats it reads</h4><ul class="pt-arch-list">${d.formats.map((x) => `<li>${x}</li>`).join("")}</ul>`;
    if (d.returns) h += `<h4>What comes out</h4><ul class="pt-arch-list">${d.returns.map((x) => `<li>${x}</li>`).join("")}</ul>`;
    if (d.iface) h += `<h4>Its interface</h4><ul class="pt-arch-list">${d.iface.map((x) => `<li><code>${esc(x[0])}</code>${x[1] ? " · " + x[1] : ""}</li>`).join("")}</ul>`;
    if (d.classes) h += `<h4>In the library</h4><ul class="pt-arch-list">${d.classes.map((x) => `<li>${x[2] ? `<a href="${esc(href(x[2]))}"><code>${esc(x[0])}</code></a>` : `<code>${esc(x[0])}</code>`}${x[1] ? " · " + x[1] : ""}</li>`).join("")}</ul>`;
    if (d.code) h += `<h4>In code</h4><pre><code>${esc(d.code)}</code></pre>`;
    if (d.bends) h += `<h4>Where the rule bends</h4><ul class="pt-arch-list">${d.bends.map((x) => `<li>${x}</li>`).join("")}</ul>`;
    if (d.tutorials) h += `<h4>See it in a tutorial</h4><ul class="pt-arch-list">${d.tutorials.map((x) => `<li><a href="${esc(href(x[1]))}">${esc(x[0])}</a></li>`).join("")}</ul>`;
    return h + `</div>`;
  }

  function openDrawer(key, from) {
    state.open = key;
    opener = from || null;
    drawer.innerHTML = detailHTML(key);
    drawer.hidden = false;
    requestAnimationFrame(() => { drawer.classList.add("is-open"); scrim.classList.add("is-open"); });
    drawer.querySelector(".pt-arch-x").addEventListener("click", closeDrawer);
    drawer.querySelector(".pt-arch-x").focus();
    applyFocus();
  }
  function closeDrawer() {
    if (!state.open) return;
    state.open = null;
    drawer.classList.remove("is-open");
    scrim.classList.remove("is-open");
    setTimeout(() => { if (!state.open) drawer.hidden = true; }, 230);
    if (opener) opener.focus();
    applyFocus();
  }
  scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

  // ---------- walkthrough ----------
  const walk = document.getElementById("pt-arch-walk");
  const walkBtn = document.getElementById("pt-arch-walk-start");
  const currentStep = () => D.walk[state.walk.m].steps[state.walk.i];

  function startWalk(m, i) {
    state.walk = { m, i };
    if (state.modality !== m) { state.modality = m; if (seg) seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === m))); layout(); }
    renderWalk();
  }
  function renderWalk() {
    if (!walk) return;
    const W = D.walk[state.walk.m], s = currentStep(), n = W.steps.length;
    const lines = W.code.split("\n");
    const on = new Set();
    (s.lines || []).forEach(([a, b]) => { for (let k = a; k <= (b || a); k++) on.add(k); });
    walk.innerHTML = `<div class="pt-arch-walk-text"><span class="pt-arch-step">${esc(W.name)} · step ${state.walk.i + 1} of ${n}</span>` +
      `<h3>${esc(s.title)}</h3><p>${s.text}</p><div class="pt-arch-walk-nav">` +
      `<button type="button" class="pt-arch-btn" data-go="-1" ${state.walk.i === 0 ? "disabled" : ""}>Back</button>` +
      `<span class="pt-arch-dots">${W.steps.map((_, k) => `<i class="${k === state.walk.i ? "on" : ""}"></i>`).join("")}</span>` +
      `<button type="button" class="pt-arch-btn is-primary" data-go="1">${state.walk.i === n - 1 ? "Finish" : "Next"}</button>` +
      `<button type="button" class="pt-arch-btn" data-go="0">Details of this part</button></div>` +
      (W.tutorial ? `<p class="pt-arch-hint" style="margin-top:10px">An excerpt: <a href="${esc(href(W.tutorial[1]))}">${esc(W.tutorial[0])}</a> has the whole code.</p>` : "") + `</div>` +
      `<pre class="pt-arch-code" aria-label="${esc(W.name)} example">${lines.map((l, k) => `<span class="${on.has(k + 1) ? "on" : ""}${/^\s*#/.test(l) ? " c" : ""}">${esc(l) || " "}</span>`).join("")}</pre>`;
    walk.hidden = false;
    walk.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => {
      const g = +b.dataset.go;
      if (g === 0) return openDrawer(currentStep().card, b);
      const j = state.walk.i + g;
      if (j >= n) { state.walk = null; walk.hidden = true; if (walkBtn) walkBtn.textContent = D.walkLabel; applyFocus(); return; }
      state.walk.i = Math.max(0, j);
      renderWalk();
    }));
    const hi = walk.querySelector(".pt-arch-code span.on");
    if (hi) hi.scrollIntoView({ block: "nearest", inline: "nearest" });
    if (walkBtn) walkBtn.textContent = "Restart the walkthrough";
    applyFocus();
  }
  if (walkBtn) {
    walkBtn.textContent = D.walkLabel;
    walkBtn.addEventListener("click", () => startWalk(state.modality === "all" ? D.modalities[0].id : state.modality, 0));
  }

  // ---------- second figure: who talks to whom, and what a swap changes ----------
  function callsFigure() {
    const fig = document.getElementById("pt-arch-calls");
    if (!fig || !D.calls) return;
    const C = D.calls;
    const bar = el("div", "pt-arch-bar");
    const segc = el("div", "pt-arch-seg");
    segc.setAttribute("role", "group");
    segc.setAttribute("aria-label", "What to swap");
    bar.append(segc);
    const note = el("p", "pt-arch-hint");
    bar.append(note);
    const svgc = svgEl("svg", { viewBox: `0 0 ${C.w} ${C.h}`, role: "img", "aria-label": C.aria });
    svgc.innerHTML = `<defs></defs>`;
    const nodes = {};
    C.edges.forEach((e) => {
      const a = C.nodes.find((n) => n.id === e.from), b = C.nodes.find((n) => n.id === e.to);
      const horiz = Math.abs(a.y - b.y) < 1;
      let x1, y1, x2, y2;
      if (horiz) { x1 = a.x + a.w; y1 = a.y + a.h / 2; x2 = b.x - 8; y2 = y1; }
      else { x1 = a.x + a.w / 2 + (e.dx || 0); y1 = a.y + a.h; x2 = b.x + b.w / 2 + (e.dx || 0); y2 = b.y - 8; }
      svgc.append(svgEl("line", { x1, y1, x2, y2 }));
      svgc.append(svgEl("polygon", { points: horiz ? `${x2},${y2 - 5} ${x2},${y2 + 5} ${x2 + 8},${y2}` : `${x2 - 5},${y2} ${x2 + 5},${y2} ${x2},${y2 + 8}` }));
      (e.label || []).forEach((t, k) => {
        const tx = horiz ? (x1 + x2) / 2 : x1 + 10, ty = horiz ? y1 - 10 - (e.label.length - 1 - k) * 15 : (y1 + y2) / 2 - (e.label.length - 1) * 7 + k * 15 + 4;
        const txt = svgEl("text", { x: tx, y: ty, class: "m", "text-anchor": horiz ? "middle" : "start" });
        txt.textContent = t;
        svgc.append(txt);
      });
    });
    C.nodes.forEach((n) => {
      const g = svgEl("g", {});
      const r = svgEl("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 12, class: "box" });
      g.append(r);
      const t = svgEl("text", { x: n.x + 14, y: n.y + 24, class: "t" }); t.textContent = n.title; g.append(t);
      (n.sub || []).forEach((s, k) => { const u = svgEl("text", { x: n.x + 14, y: n.y + 44 + k * 16, class: "s" }); u.textContent = s; g.append(u); });
      const tag = svgEl("text", { x: n.x + 14, y: n.y + n.h - 12, class: "tag" });
      g.append(tag);
      svgc.append(g);
      nodes[n.id] = { r, tag };
    });
    const figure = el("figure", "pt-arch-fig");
    figure.append(svgc);
    const cap = el("figcaption", null, C.caption);
    figure.append(cap);
    fig.append(bar, figure);
    function setSwap(id) {
      const s = C.swaps.find((x) => x.id === id);
      segc.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.s === id)));
      C.nodes.forEach((n) => {
        const on = s.swap.includes(n.id);
        nodes[n.id].r.setAttribute("class", "box " + (s.id === "none" ? "" : on ? "swap" : "keep"));
        nodes[n.id].tag.setAttribute("class", "tag " + (on ? "swap" : "keep"));
        nodes[n.id].tag.textContent = s.id === "none" ? "" : on ? "you change this" : "unchanged";
      });
      note.innerHTML = s.note;
    }
    C.swaps.forEach((s) => {
      const b = el("button", null, esc(s.name));
      b.type = "button";
      b.dataset.s = s.id;
      b.addEventListener("click", () => setSwap(s.id));
      segc.append(b);
    });
    setSwap(C.swaps[0].id);
  }

  build();
  callsFigure();
  layout();
  if (window.ResizeObserver) new ResizeObserver(() => layout()).observe(host);
  else window.addEventListener("resize", layout);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout);
})();
