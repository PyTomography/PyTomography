/* PyTomography docs: opens the 3D viewer from a tutorial page's viewer block or a gallery card's "View in 3D" button.
   The viewer (pt-viewer.js and pt-viewer.css, next to this file) and the images load only when a reader opens it. */
(function () {
  'use strict';
  const BASE = new URL('.', document.currentScript.src).href;
  const VER = document.currentScript.dataset.version ? '?v=' + document.currentScript.dataset.version : '';
  let loading = null;
  function load() {
    if (window.PTViewer) return Promise.resolve();
    if (!loading) loading = new Promise((ok, fail) => {
      const css = document.createElement('link');
      css.rel = 'stylesheet'; css.href = BASE + 'pt-viewer.css' + VER;
      document.head.appendChild(css);
      const js = document.createElement('script');
      js.src = BASE + 'pt-viewer.js' + VER;
      js.onload = ok;
      js.onerror = () => { loading = null; fail(new Error('The viewer did not load. Check your connection and try again.')); };
      document.head.appendChild(js);
    });
    return loading;
  }
  const narrow = () => window.matchMedia && matchMedia('(max-width: 700px)').matches;

  // Full screen, over the page: the gallery, and every page on a phone
  async function openOver(manifest, title, opener, msg) {
    try { await load(); } catch (e) { if (msg) msg.textContent = e.message; return; }
    const el = document.createElement('div');
    el.setAttribute('role', 'dialog'); el.setAttribute('aria-modal', 'true'); el.setAttribute('aria-label', title + ', 3D viewer');
    document.body.appendChild(el);
    PTViewer.mount(el, {manifest, title, maximized: true, noMaximize: true,
      onClose: api => { api.destroy(); el.remove(); if (opener) opener.focus(); }});
    const close = el.querySelector('.ptv-actions button');
    if (close) close.focus();
  }

  // Tutorial pages: the viewer opens in place of the block, and Close puts the block back
  document.querySelectorAll('.ptv-block').forEach(block => {
    const btn = block.querySelector('.ptv-open'), msg = block.querySelector('.ptv-msg');
    btn.addEventListener('click', async () => {
      const {manifest, title} = block.dataset;
      if (narrow()) return openOver(manifest, title, btn, msg);
      btn.disabled = true; msg.textContent = '';
      try { await load(); } catch (e) { btn.disabled = false; msg.textContent = e.message; return; }
      const host = document.createElement('div');
      block.appendChild(host);
      btn.hidden = true; btn.disabled = false;
      PTViewer.mount(host, {manifest, title, onClose: api => { api.destroy(); host.remove(); btn.hidden = false; btn.focus(); }});
      // the viewer is as tall as the window: bring all of it into view, below the site's header
      const still = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      host.scrollIntoView({block: 'start', behavior: still ? 'auto' : 'smooth'});
    });
  });

  // Gallery: a "View in 3D" button on each card whose tutorial has images
  const index = document.getElementById('ptv-index');
  if (!index) return;
  let cards = {};
  try { cards = JSON.parse(index.textContent); } catch (_) { return; }
  document.querySelectorAll('a.pt-card').forEach(card => {
    const name = (card.getAttribute('href') || '').split('/').pop().replace(/\.html.*$/, '');
    const t = cards[name];
    if (!t) return;
    const wrap = document.createElement('div');
    wrap.className = 'ptv-cardwrap';
    card.replaceWith(wrap);
    wrap.appendChild(card);
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'ptv-card3d'; b.textContent = 'View in 3D';
    b.setAttribute('aria-label', `View ${t.title} in 3D`);
    b.addEventListener('click', () => openOver(t.manifest, t.title, b, null));
    wrap.appendChild(b);
    // the gallery's filters hide the card; hide its button with it
    const sync = () => { wrap.hidden = card.hidden; };
    sync();
    new MutationObserver(sync).observe(card, {attributes: true, attributeFilter: ['hidden']});
  });
})();
