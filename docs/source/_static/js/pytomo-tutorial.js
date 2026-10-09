// Tutorial pages: this page's headings, nested under the current tutorial in the pinned section navigation on the
// left, with the one in view highlighted. It replaces the right-hand "On this page" sidebar, which is off on notebook
// pages so code and the 3D viewer get the width. Loaded on notebook pages only (_ext/pytomo_viewer.py).
document.addEventListener("DOMContentLoaded", () => {
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
  // highlight the heading whose section is in view
  if (!("IntersectionObserver" in window)) return;
  const visible = new Set();
  const mark = () => {
    const sections = [...links.keys()];
    const top = sections.find((s) => visible.has(s)) || null;
    links.forEach((a, s) => a.setAttribute("aria-current", String(s === top)));
  };
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => (e.isIntersecting ? visible.add(e.target) : visible.delete(e.target)));
    mark();
  }, {rootMargin: "-80px 0px -55% 0px"});
  links.forEach((_, s) => io.observe(s));
});
