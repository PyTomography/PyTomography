// Tutorial pages (loaded on notebook pages only, by _ext/pytomo_viewer.py):
// - this page's headings, nested under the current tutorial in the pinned section navigation, the one in view
//   highlighted (the right-hand "On this page" sidebar is off on notebook pages, so the tutorial gets the width);
// - the header's data popup (_ext/tutorial_page.py embeds the datasets as JSON), and Copy page as Markdown on the tabs;
// - code lines that wrap under their own start, and outputs without matplotlib's return values.
(() => {
  "use strict";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

  // ---------- this page's headings in the section navigation ----------
  function pageToc() {
    const nav = document.querySelector(".bd-sidebar-primary .bd-docs-nav");
    const here = nav && nav.querySelector("li.current.active > a, li.current > a.current");
    const article = document.querySelector(".bd-article");
    if (!here || !article) return;
    const heads = [...article.querySelectorAll("section > h2, section > h3")].filter((h) => h.parentElement.id);
    if (!heads.length) return;
    const list = document.createElement("ul");
    list.className = "pt-pagetoc";
    list.setAttribute("aria-label", "On this page");
    const links = new Map();
    heads.forEach((h) => {
      const copy = h.cloneNode(true);
      copy.querySelectorAll(".headerlink").forEach((a) => a.remove());
      const li = document.createElement("li");
      li.className = h.tagName === "H3" ? "pt-l3" : "pt-l2";
      const a = document.createElement("a");
      a.href = "#" + h.parentElement.id;
      a.textContent = copy.textContent.trim();
      li.appendChild(a);
      list.appendChild(li);
      links.set(h.parentElement, a);
    });
    here.insertAdjacentElement("afterend", list);
    if (!("IntersectionObserver" in window)) return;
    const visible = new Set();
    const mark = () => {
      const top = [...links.keys()].find((s) => visible.has(s)) || null;
      links.forEach((a, s) => a.setAttribute("aria-current", String(s === top)));
    };
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? visible.add(e.target) : visible.delete(e.target)));
      mark();
    }, {rootMargin: "-80px 0px -55% 0px"});
    links.forEach((_, s) => io.observe(s));
  }

  // ---------- the data popup ----------
  const DATA_ICON = '<svg class="pt-i" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true"><ellipse cx="10" cy="5" rx="6" ry="2.4"/><path d="M4 5v10c0 1.3 2.7 2.4 6 2.4s6-1.1 6-2.4V5"/>' +
    '<path d="M4 10c0 1.3 2.7 2.4 6 2.4s6-1.1 6-2.4"/></svg>';
  function copyButton(btn) {
    btn.addEventListener("click", () => {
      const done = (t) => { btn.textContent = t; setTimeout(() => { btn.textContent = "Copy"; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(btn.dataset.copy).then(() => done("Copied"), () => done("Copy failed"));
    });
  }
  function dataPopup(tut) {
    const island = tut.querySelector("script.pt-data"), panel = tut.querySelector(".pt-tut-panel");
    if (!island || !panel) return;
    let list = [];
    try { list = JSON.parse(island.textContent); } catch (_) { return; }
    const byKey = {};
    list.forEach((e) => { byKey[e.key] = e; });
    const page = panel.dataset.page || "";
    const pop = document.createElement("div");
    pop.className = "pt-dpop"; pop.id = "pt-dpop"; pop.hidden = true; pop.tabIndex = -1;
    pop.setAttribute("role", "dialog");
    panel.appendChild(pop);
    let chip = null;
    const render = (e) => {
      const size = e.disk ? `${esc(e.download)} to download, ${esc(e.disk)} on disk` : esc(e.download);
      const rows = [["Source", e.url ? `<a href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.source)}</a>` : esc(e.source)]];
      if (e.available !== false && e.download) rows.push(["Size", size]);
      rows.push(["Licence", esc(e.licence)]);
      if (e.cite) rows.push(["Cite", esc(e.cite)]);
      if (e.used_by && e.used_by.length) rows.push(["Also in", e.used_by.map((u) => `<a href="${esc(u.notebook)}.html">${esc(u.title)}</a>`).join(", ")]);
      let get;
      if (e.available === false) get = `<p class="pt-dpop-pending">Not downloadable yet. ${esc(e.note || "")}</p>`;
      else if (e.fetch) get = `<div class="pt-dpop-get"><code>${esc(e.fetch)}</code><button type="button" class="pt-dpop-copy" data-copy="${esc(e.fetch)}">Copy</button></div>` +
        `<p class="pt-dpop-hint">${e.needs ? `It needs <code>${esc(e.needs)}</code>. ` : ""}The first code cell runs this. ` +
        "It downloads the data once, into the folder set by <code>PYTOMOGRAPHY_DATA</code>.</p>";
      else get = `<p class="pt-dpop-steps">${esc(e.steps || "")}</p>`;
      return `<div class="pt-dpop-head"><p class="pt-dpop-title">${DATA_ICON}<code>${esc(e.key)}</code></p>` +
        '<button type="button" class="pt-dpop-x" aria-label="Close">&times;</button></div>' +
        `<p class="pt-dpop-what">${esc(e.title)}</p><p class="pt-dpop-k">How to get it</p>${get}` +
        `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
        `<p class="pt-dpop-foot"><a href="${esc(page)}#${esc(e.anchor)}">All the tutorial data</a></p>`;
    };
    const close = (focus) => {
      if (!chip) return;
      pop.hidden = true; chip.setAttribute("aria-expanded", "false");
      if (focus) chip.focus();
      chip = null;
    };
    const open = (c) => {
      const e = byKey[c.dataset.key];
      if (!e) return;
      if (chip === c) return close(true);
      close(false);
      pop.innerHTML = render(e);
      pop.setAttribute("aria-label", e.key);
      pop.hidden = false;
      pop.style.left = ""; pop.style.top = "";
      if (getComputedStyle(pop).position === "absolute") {   // under its button, inside the panel
        const r = panel.getBoundingClientRect(), b = c.getBoundingClientRect();
        pop.style.left = Math.max(0, Math.min(b.left - r.left, r.width - pop.offsetWidth)) + "px";
        pop.style.top = (b.bottom - r.top + 8) + "px";
      }
      c.setAttribute("aria-expanded", "true"); chip = c;
      pop.querySelector(".pt-dpop-x").addEventListener("click", () => close(true));
      pop.querySelectorAll("[data-copy]").forEach(copyButton);
      pop.focus({preventScroll: true});
    };
    tut.querySelectorAll("button.pt-dchip").forEach((c) => c.addEventListener("click", () => open(c)));
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && chip) { ev.stopPropagation(); close(true); } }, true);
    document.addEventListener("click", (ev) => { if (chip && !pop.contains(ev.target) && !ev.target.closest(".pt-dchip")) close(false); });
  }

  // ---------- Copy page as Markdown (made by js/pytomo.js) joins the tabs ----------
  function copyMarkdown(tut) {
    const slot = tut.querySelector(".pt-tb-end"), b = document.querySelector(".pt-copy-md");
    if (slot && b && !slot.contains(b)) slot.appendChild(b);
  }

  // ---------- code: each line in its own span, so a wrapped line continues under its own start ----------
  function lines(pre) {
    const out = [[]];
    for (const n of Array.from(pre.childNodes)) {
      if (n.nodeType === 1 && n.children.length) return;    // nested markup (highlighted lines): leave the block as it is
      n.textContent.split("\n").forEach((p, i) => {
        if (i) out.push([]);
        if (!p) return;
        let x;
        if (n.nodeType === 3) x = document.createTextNode(p);
        else { x = n.cloneNode(false); x.textContent = p; }
        out[out.length - 1].push(x);
      });
    }
    const frag = document.createDocumentFragment();
    out.forEach((nodes, i) => {
      const last = i === out.length - 1;
      if (last && !nodes.length) return;                    // the newline that ends the block stays a newline
      const ln = document.createElement("span");
      ln.className = "pt-ln";
      const text = nodes.map((x) => x.textContent).join("");
      ln.style.setProperty("--h", (text.length - text.replace(/^ +/, "").length + 4) + "ch");
      nodes.forEach((x) => ln.appendChild(x));
      frag.appendChild(ln);
      if (!last) frag.appendChild(document.createTextNode("\n"));
    });
    pre.replaceChildren(frag);
  }

  // ---------- outputs: hide matplotlib's return values, e.g. <matplotlib.colorbar.Colorbar at 0x7f...> ----------
  const REPR = /^\s*(?:\[?\s*(?:<[\w.]+(?: object)? at 0x[0-9a-f]+>\s*,?\s*)+\]?|<Figure size \d+x\d+ with \d+ Axes>|Text\(.*\))\s*$/;
  function quietOutputs() {
    document.querySelectorAll(".bd-article div.cell_output > .output.text_plain").forEach((o) => {
      const pre = o.querySelector("pre");                   // the text only, not the copy button's label
      if (pre && REPR.test(pre.textContent)) o.hidden = true;
    });
    document.querySelectorAll(".bd-article div.cell_output").forEach((c) => { if (!Array.from(c.children).some((x) => !x.hidden)) c.hidden = true; });
  }

  // after js/pytomo.js, which adds Copy page as Markdown when the page has loaded
  document.addEventListener("DOMContentLoaded", () => {
    pageToc();
    const tut = document.querySelector(".pt-tut");
    if (tut) { dataPopup(tut); copyMarkdown(tut); }
    document.querySelectorAll('.bd-article div[class^="highlight-"] > div.highlight > pre').forEach(lines);
    quietOutputs();
  });
})();
