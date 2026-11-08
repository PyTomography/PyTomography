// PyTomography docs: copy buttons, gallery and board filters, "Copy page as Markdown".
document.addEventListener("DOMContentLoaded", () => {
  // Copy buttons on the landing page
  document.querySelectorAll(".pt-copy[data-copy]").forEach((btn) => {
    btn.addEventListener("click", () => {
      navigator.clipboard?.writeText(btn.dataset.copy).then(() => {
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = "Copy"), 1500);
      }).catch(() => {});
    });
  });

  // Filters: chips with data-key/data-value filter items carrying data-<key> attributes
  const setupFilters = (root, itemSelector, groupSelector) => {
    const state = {};
    const chips = [...root.querySelectorAll(".pt-chip")];
    const count = root.querySelector(".pt-count");
    const items = [...root.querySelectorAll(itemSelector)];
    const apply = () => {
      let shown = 0;
      items.forEach((item) => {
        const ok = Object.entries(state).every(([k, v]) => v === "All" || item.dataset[k] === v || item.dataset[k] === "Any");
        item.hidden = !ok;
        if (ok) shown++;
      });
      if (groupSelector) root.querySelectorAll(groupSelector).forEach((g) => {
        g.hidden = !g.querySelector(itemSelector + ":not([hidden])");
      });
      if (count) count.textContent = `${shown} of ${items.length}`;
      const empty = root.querySelector(".pt-empty");
      if (empty) empty.hidden = shown > 0;
    };
    chips.forEach((chip) => {
      state[chip.dataset.key] = state[chip.dataset.key] || "All";
      chip.addEventListener("click", () => {
        state[chip.dataset.key] = chip.dataset.value;
        chips.filter((c) => c.dataset.key === chip.dataset.key)
          .forEach((c) => c.setAttribute("aria-pressed", c === chip));
        apply();
      });
    });
    apply();
  };
  document.querySelectorAll("[data-gallery]").forEach((g) => setupFilters(g, ".pt-card", ".pt-gsection"));
  document.querySelectorAll("[data-board]").forEach((b) => setupFilters(b, ".pt-bcard", null));

  // Notebook / Script toggle on tutorial pages; the choice is remembered across tutorials
  const viewSwitch = document.querySelector(".pt-viewswitch");
  if (viewSwitch) {
    const setView = (view) => {
      document.body.classList.toggle("pt-mode-script", view === "script");
      viewSwitch.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.view === view));
    };
    let saved = "notebook";
    try { saved = localStorage.getItem("pt-tutorial-view") || "notebook"; } catch (e) {}
    setView(saved);
    viewSwitch.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      setView(b.dataset.view);
      try { localStorage.setItem("pt-tutorial-view", b.dataset.view); } catch (e) {}
    });
  }

  // Copy page as Markdown: fetch the page source; notebooks are converted to Markdown.
  const meta = document.querySelector('meta[name="pt-source"]');
  const h1 = document.querySelector(".bd-article h1");
  if (meta && h1 && !document.querySelector(".pt-landing")) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pt-copy-md";
    btn.textContent = "Copy page as Markdown";
    h1.before(btn);
    const notebookToMarkdown = (nb) =>
      nb.cells.map((c) => {
        const src = Array.isArray(c.source) ? c.source.join("") : c.source;
        return c.cell_type === "code" ? "```python\n" + src + "\n```" : src;
      }).join("\n\n");
    btn.addEventListener("click", async () => {
      let text;
      try {
        const res = await fetch(meta.content);
        text = await res.text();
        if (meta.content.includes(".ipynb")) text = notebookToMarkdown(JSON.parse(text));
      } catch (e) {
        text = document.querySelector(".bd-article").innerText;
      }
      text = `<!-- Source: ${location.href} -->\n\n` + text;
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
      } catch (e) {
        btn.textContent = "Copy failed";
      }
      setTimeout(() => (btn.textContent = "Copy page as Markdown"), 1800);
    });
  }
});
