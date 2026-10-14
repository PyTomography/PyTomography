/* The Architecture page's overview: the whole library as one diagram, with the reconstruction's three layers nested so
   encapsulation shows (the algorithm holds a likelihood, which holds a system matrix), input on the left and output on
   the right. Every box and name opens a popup listing what the library has of that kind (window.PT_CATALOG), which
   the reader can step through. Also the page's tabs. */
(function () {
  "use strict";
  const CAT = window.PT_CATALOG;
  const host = document.getElementById("pt-arch-overview");
  if (!CAT || !host) return;
  const NS = "http://www.w3.org/2000/svg";
  const NARROW = 980;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const svgEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const wbr = (t) => esc(t).replace(/([a-z0-9])([A-Z])/g, "$1<wbr>$2").replace(/([A-Z])([A-Z][a-z])/g, "$1<wbr>$2").replace(/([._])/g, "$1<wbr>");

  // ---------- the diagram ----------
  // a chip names one item of a category; a node is a box that opens its category
  const chip = (cat, name, label) => {
    const i = CAT[cat].items.findIndex((x) => x.name === name);
    return `<button type="button" class="pto-chip" data-cat="${cat}" data-i="${Math.max(0, i)}">${wbr(label || name)}</button>`;
  };
  const head = (cat, title, kicker, n) =>
    `<div class="pto-head"><button type="button" class="pto-title" data-cat="${cat}" data-i="0"><b>${esc(title)}</b>` +
    `<span class="pto-count">${n ?? CAT[cat].items.length}<i aria-hidden="true">›</i></span></button>` +
    (kicker ? `<span class="pto-kicker">${esc(kicker)}</span>` : "") + `</div>`;
  const node = (id, cat, title, kicker, chips, extra = "") =>
    `<div class="pto-node${extra}" id="pto-${id}" data-cat="${cat}">${head(cat, title, kicker)}` +
    (chips.length ? `<div class="pto-chips">${chips.join("")}</div>` : "") + `</div>`;

  host.innerHTML = `
  <div class="pto">
    <div class="pto-col pto-in">
      <p class="pto-eyebrow">Input</p>
      ${node("files", "files", "Data files", "from scanners and simulators", [chip("files", "DICOM NM projections", "DICOM"), chip("files", "SIMIND interfile", "SIMIND"), chip("files", "GATE (ROOT and .mac)", "GATE"), chip("files", "PETSIRD"), chip("files", "GE Discovery MI (HDF5)", "GE HDF5"), chip("files", "DICOM-CT-PD")])}
      ${node("readers", "readers", "Readers", "pytomography.io", [chip("readers", "io.SPECT.dicom", "SPECT.dicom"), chip("readers", "io.SPECT.simind", "SPECT.simind"), chip("readers", "io.PET.gate", "PET.gate"), chip("readers", "io.PET.petsird", "PET.petsird"), chip("readers", "io.CT.dicom_ct_pd", "CT.dicom_ct_pd")])}
      ${node("metadata", "metadata", "Standard objects", "pytomography.metadata + tensors", [chip("metadata", "ObjectMeta"), chip("metadata", "SPECTProjMeta"), chip("metadata", "PETLMProjMeta"), chip("metadata", "CTGen3ProjMeta"), chip("metadata", "Tensors", "tensors")])}
    </div>
    <div class="pto-engine">
      <p class="pto-eyebrow">Reconstruction</p>
      <div class="pto-layer pto-l1" id="pto-alg" data-cat="algorithm">
        ${head("algorithm", "Algorithm", "pytomography.algorithms · the update rule")}
        <div class="pto-chips">${["OSEM", "BSREM", "OSMAPOSL", "KEM", "SART", "FilteredBackProjection", "DIPRecon"].map((n) => chip("algorithm", n, n === "FilteredBackProjection" ? "FBP" : n)).join("")}</div>
        <div class="pto-l1-body">
          <div class="pto-layer pto-l2" id="pto-lik" data-cat="likelihood">
            <span class="pto-iface" title="The only call the algorithm makes">compute_gradient(f, m)</span>
            ${head("likelihood", "Likelihood", "pytomography.likelihoods · the noise model")}
            <div class="pto-chips">${chip("likelihood", "PoissonLogLikelihood", "Poisson")}${chip("likelihood", "NegativeMSELikelihood", "least squares")}${chip("likelihood", "SARTWeightedNegativeMSELikelihood", "weighted least squares")}</div>
            <div class="pto-pills" id="pto-gs">
              <button type="button" class="pto-pill" data-cat="data" data-i="0"><b>g</b> measured data</button>
              <button type="button" class="pto-pill" data-cat="data" data-i="1"><b>s</b> additive term: scatter, randoms</button>
            </div>
            <div class="pto-layer pto-l3" id="pto-sm" data-cat="sysmat">
              <span class="pto-iface" title="The only calls the likelihood makes">forward(f) &nbsp;·&nbsp; backward(g)</span>
              ${head("sysmat", "System matrix H", "pytomography.projectors · all of the geometry and physics")}
              <div class="pto-pills" id="pto-metas">
                <button type="button" class="pto-pill" data-cat="metadata" data-i="0"><b>object_meta</b> image grid</button>
                <button type="button" class="pto-pill" data-cat="metadata" data-i="2"><b>proj_meta</b> data geometry</button>
              </div>
              <div class="pto-chips">${chip("sysmat", "SPECTSystemMatrix")}${chip("sysmat", "StarGuideSystemMatrix")}${chip("sysmat", "PETLMSystemMatrix")}${chip("sysmat", "PETSinogramSystemMatrix")}${chip("sysmat", "CTGen3SystemMatrix")}${chip("sysmat", "Your own", "your own")}</div>
              <div class="pto-chain" id="pto-chain">
                ${node("obj2obj", "obj2obj", "Object transforms", "attenuation, PSF", [], " pto-sub")}
                <span class="pto-chain-arrow" aria-hidden="true"></span>
                ${node("proj", "projector", "Projector", "rays, rotation, TOF", [], " pto-sub pto-core")}
                <span class="pto-chain-arrow" aria-hidden="true"></span>
                ${node("proj2proj", "proj2proj", "Projection transforms", "masks, additive", [], " pto-sub")}
              </div>
              <p class="pto-note">forward: left to right · backward: right to left, each step transposed</p>
            </div>
          </div>
          <div class="pto-side">
            ${node("prior", "prior", "Prior", "optional · a penalty V(f)", [chip("prior", "RelativeDifferencePrior", "relative difference"), chip("prior", "QuadraticPrior", "quadratic")], " pto-opt")}
            ${node("callback", "callback", "Callbacks", "optional · after each subiteration", [chip("callback", "DataStorageCallback")], " pto-opt")}
          </div>
        </div>
      </div>
    </div>
    <div class="pto-col pto-out">
      <p class="pto-eyebrow">Output</p>
      ${node("recon", "output", "Reconstruction f", "a tensor on object_meta's grid", [chip("output", "Patient frames", "patient frame"), chip("output", "Uncertainty", "uncertainty")])}
      ${node("files-out", "output", "Saved and shown", "pytomography.io", [chip("output", "save_dicom"), chip("output", "save_nifti"), chip("output", "Plots and the 3D viewer", "3D viewer")])}
    </div>
    <svg class="pto-svg" aria-hidden="true"></svg>
  </div>
  <p class="pto-legend"><span class="pto-legend-iface">method</span> on a border: the only way into that layer &nbsp;·&nbsp; dashed: optional &nbsp;·&nbsp; click any box or name to browse</p>`;

  const pto = host.querySelector(".pto");
  const svg = host.querySelector(".pto-svg");

  // arrows between the columns and into the nested layers
  function arrows() {
    svg.textContent = "";
    const narrow = host.clientWidth < NARROW;
    pto.classList.toggle("is-narrow", narrow);
    const P = pto.getBoundingClientRect();
    const R = (id) => { const r = document.getElementById(id).getBoundingClientRect(); return { l: r.left - P.left, r: r.right - P.left, t: r.top - P.top, b: r.bottom - P.top, cx: (r.left + r.right) / 2 - P.left, cy: (r.top + r.bottom) / 2 - P.top }; };
    const draw = (x1, y1, x2, y2, label, opt = {}) => {
      const g = svgEl("g", { class: opt.cls || "" });
      const vertical = Math.abs(x2 - x1) < 2 || opt.vertical;
      let d;
      if (vertical) d = `M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2 - 8}`;
      else { const mx = x1 + (x2 - x1) * (opt.bend ?? 0.5); d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2 - 8},${y2}`; }
      g.append(svgEl("path", { d }));
      g.append(svgEl("polygon", { points: vertical ? `${x2 - 5},${y2 - 8} ${x2 + 5},${y2 - 8} ${x2},${y2}` : `${x2 - 8},${y2 - 5} ${x2 - 8},${y2 + 5} ${x2},${y2}` }));
      svg.append(g);
      if (label) {
        const lx = opt.lx ?? (vertical ? x1 + 10 : (x1 + x2) / 2), ly = opt.ly ?? (vertical ? (y1 + y2) / 2 + 4 : Math.min(y1, y2) - 8);
        const t = svgEl("text", { x: lx, y: ly, "text-anchor": vertical ? "start" : "middle" });
        t.textContent = label;
        svg.append(t);
        const bb = t.getBBox();
        svg.insertBefore(svgEl("rect", { x: bb.x - 4, y: bb.y - 1, width: bb.width + 8, height: bb.height + 2, rx: 4, class: "lab" }), t);
      }
    };
    const files = R("pto-files"), readers = R("pto-readers"), meta = R("pto-metadata");
    const recon = R("pto-recon"), saved = R("pto-files-out"), alg = R("pto-alg");
    const sm = R("pto-sm"), gs = R("pto-gs"), metas = R("pto-metas");
    draw(files.cx, files.b + 2, readers.cx, readers.t, "read by");
    draw(readers.cx, readers.b + 2, meta.cx, meta.t, "return");
    draw(recon.cx, recon.b + 2, saved.cx, saved.t, "save, show");
    if (narrow) {
      draw(meta.cx, meta.b + 2, meta.cx, alg.t - 26, "into H and the likelihood");
      draw(alg.cx, alg.b + 2, recon.cx, recon.t - 26, "returns f");
      return;
    }
    // the metadata configures H; the data fills the likelihood; the algorithm returns f
    // the pills they point at name what flows: g and s fill the likelihood, the metadata configures H
    draw(meta.r + 2, meta.t + 24, gs.l, gs.cy, null, { bend: 0.45 });
    draw(meta.r + 2, meta.t + 52, metas.l, metas.cy, null, { bend: 0.45 });
    draw(alg.r + 2, recon.cy, recon.l, recon.cy, "returns f");
    void sm;
  }

  // ---------- the popup: browse one category ----------
  const pop = document.createElement("div");
  pop.className = "pto-pop";
  pop.hidden = true;
  pop.innerHTML = `<div class="pto-scrim"></div><div class="pto-dialog" role="dialog" aria-modal="true" aria-labelledby="pto-pop-title"></div>`;
  document.body.append(pop);
  const dialog = pop.querySelector(".pto-dialog");
  let cur = null, opener = null;

  function detail(it) {
    let h = `<h3>${esc(it.name)}</h3>`;
    if (it.tags) h += `<p class="pto-tags">${it.tags.map((t) => `<span>${esc(t)}</span>`).join("")}</p>`;
    if (it.what) h += `<p class="pto-what">${esc(it.what)}</p>`;
    if (it.sig) h += `<h4>Signature</h4><pre><code>${esc(it.sig)}</code></pre>`;
    if (it.code) h += `<h4>In code</h4><pre><code>${esc(it.code)}</code></pre>`;
    const links = [];
    if (it.api) links.push(`<a href="${esc(it.api)}">API reference ›</a>`);
    (it.tut || []).forEach(([n, u]) => links.push(`<a href="${esc(u)}">${esc(n)} tutorial ›</a>`));
    if (links.length) h += `<p class="pto-links">${links.join("")}</p>`;
    return h;
  }
  function render() {
    const C = CAT[cur.cat], it = C.items[cur.i], n = C.items.length;
    dialog.innerHTML =
      `<header><div><span class="pto-kicker">${esc(C.kicker)}</span><h2 id="pto-pop-title">${esc(C.title)} <small>${n}</small></h2>` +
      `<p>${esc(C.blurb)}</p></div><button type="button" class="pto-x">Close</button></header>` +
      `<div class="pto-pop-body"><nav class="pto-list" aria-label="${esc(C.title)}">` +
      C.items.map((x, k) => `<button type="button" data-k="${k}" class="${k === cur.i ? "on" : ""}" aria-current="${k === cur.i}">${wbr(x.name)}` +
        `${x.tags ? `<small>${esc(x.tags.join(" · "))}</small>` : ""}</button>`).join("") +
      `</nav><select class="pto-select" aria-label="${esc(C.title)}">${C.items.map((x, k) => `<option value="${k}" ${k === cur.i ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select>` +
      `<article class="pto-detail">${detail(it)}</article></div>` +
      `<footer><button type="button" class="pto-step" data-d="-1" ${cur.i === 0 ? "disabled" : ""}>‹ Previous</button>` +
      `<span>${cur.i + 1} of ${n}</span><button type="button" class="pto-step" data-d="1" ${cur.i === n - 1 ? "disabled" : ""}>Next ›</button></footer>`;
    dialog.querySelector(".pto-x").addEventListener("click", close);
    dialog.querySelectorAll(".pto-list button").forEach((b) => b.addEventListener("click", () => go(+b.dataset.k)));
    dialog.querySelector(".pto-select").addEventListener("change", (e) => go(+e.target.value));
    dialog.querySelectorAll(".pto-step").forEach((b) => b.addEventListener("click", () => go(cur.i + +b.dataset.d, true)));
    const on = dialog.querySelector(".pto-list .on");
    if (on) on.scrollIntoView({ block: "nearest" });
  }
  function go(i, keepFocus) {
    const n = CAT[cur.cat].items.length;
    cur.i = Math.max(0, Math.min(n - 1, i));
    render();
    const f = keepFocus ? dialog.querySelector(`.pto-step[data-d="${i > cur.i ? 1 : -1}"]:not([disabled])`) || dialog.querySelector(".pto-list .on") : dialog.querySelector(".pto-list .on");
    if (f) f.focus();
  }
  function open(cat, i, from) {
    cur = { cat, i: i || 0 };
    opener = from || null;
    render();
    pop.hidden = false;
    document.documentElement.classList.add("pto-lock");
    (dialog.querySelector(".pto-list .on") || dialog.querySelector(".pto-x")).focus();
  }
  function close() {
    pop.hidden = true;
    document.documentElement.classList.remove("pto-lock");
    if (opener) opener.focus();
  }
  pop.querySelector(".pto-scrim").addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (pop.hidden) return;
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if ((e.key === "ArrowDown" || e.key === "ArrowRight") && e.target.tagName !== "SELECT") { e.preventDefault(); go(cur.i + 1); }
    else if ((e.key === "ArrowUp" || e.key === "ArrowLeft") && e.target.tagName !== "SELECT") { e.preventDefault(); go(cur.i - 1); }
  });
  host.addEventListener("click", (e) => {
    const b = e.target.closest("[data-cat]");
    if (!b || !host.contains(b)) return;
    if (b.matches("button")) return open(b.dataset.cat, +(b.dataset.i || 0), b);
    // a click on a box's background opens its category
    if (b.matches(".pto-node, .pto-layer") && !e.target.closest("button")) open(b.dataset.cat, 0, b.querySelector(".pto-title"));
  });
  // a box's background lights up only when the pointer is on it, not on a box inside it
  host.addEventListener("mouseover", (e) => {
    host.querySelectorAll(".is-hover").forEach((x) => x.classList.remove("is-hover"));
    const b = e.target.closest(".pto-node, .pto-layer");
    if (b) b.classList.add("is-hover");
  });
  host.addEventListener("mouseleave", () => host.querySelectorAll(".is-hover").forEach((x) => x.classList.remove("is-hover")));

  // ---------- tabs ----------
  const tabs = [...document.querySelectorAll(".pt-arch-tabs [role=tab]")];
  function show(id, push) {
    tabs.forEach((t) => {
      const on = t.getAttribute("aria-controls") === id;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (push) history.replaceState(null, "", "#" + id.replace(/^tab-/, ""));
    window.dispatchEvent(new Event("resize"));
  }
  tabs.forEach((t, k) => {
    t.addEventListener("click", () => show(t.getAttribute("aria-controls"), true));
    t.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      const n = tabs[(k + d + tabs.length) % tabs.length];
      n.focus();
      show(n.getAttribute("aria-controls"), true);
    });
  });
  const fromHash = () => { const id = "tab-" + location.hash.slice(1); if (document.getElementById(id) && tabs.some((t) => t.getAttribute("aria-controls") === id)) show(id, false); };
  window.addEventListener("hashchange", fromHash);
  fromHash();

  arrows();
  if (window.ResizeObserver) new ResizeObserver(() => arrows()).observe(host);
  window.addEventListener("resize", arrows);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(arrows);
})();
