/* The memory calculator on the "Memory and speed" page: its inputs and results. The arithmetic is in pytomo-memory.js
 * (the memory model) and memory-calc.js (budgets, plans, suggestions, links); this file reads inputs and draws. It
 * loads its four data files from _static/memory/ (no other network access), shows a loading bar held at the
 * calculator's size until all have arrived, then draws the calculator whole. */
(function () {
  'use strict';
  var root = document.getElementById('pt-mem');
  if (!root) { return; }
  var M = window.PyTomoMemory, Calc = window.PyTomoMemoryCalc;
  var base = root.getAttribute('data-base') || '_static/memory/';
  var FILES = [['model.json', 'memory model'], ['pet_scanners.json', 'PET scanners'], ['gpus.json', 'GPUs'], ['reference_cases.json', 'tutorials']];
  var MODALITIES = ['PET', 'CT', 'SPECT'];
  // The inputs shown, as cells [label, parameters, separator]; a model's other parameters go under Advanced.
  var LAYOUT = {
    PET: [['Image', ['Lx', 'Ly', 'Lz'], '×'], ['Crystals × rings', ['crystals_per_ring', 'rings'], '×'], ['TOF', ['tof', 'tof_bins']],
          ['Events', ['events']], ['Algorithm', ['algorithm']], ['Prior', ['prior']], ['Scatter from', ['sss_input']]],
    CT: [['Image', ['Lx', 'Ly', 'Lz'], '×'], ['Views', ['views']], ['Detector', ['det_cols', 'det_rows'], '×'], ['Data in', ['ct_data']]],
    SPECT: [['Image', ['Lx', 'Ly', 'Lz'], '×'], ['Views', ['views']], ['Angles', ['sg_angles']], ['Views/angle', ['sg_views_per_angle']],
            ['PSF', ['psf']], ['Radii', ['n_radii']], ['Algorithm', ['algorithm']], ['Prior', ['prior']], ['KEM size', ['kem_size']], ['Beds', ['n_beds']]]
  };
  var SCANNER_GROUPS = [['PET/CT', 'PET/CT'], ['PET/MR', 'PET/MR'], ['total-body', 'Long axial FOV'], ['brain', 'Brain'], ['preclinical', 'Preclinical']];
  var KEEP_BY_DEFAULT = ['n_subsets'];
  var SCOPES = {held: 'held', subset: 'per subset', chunk: 'per chunk', overhead: 'overhead'};

  var data = {}, state = null, ui = {}, timer = null;

  // ---------------------------------------------------------------------------------------------- small helpers
  function el(tag, attrs, children) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) { return; }
      if (k === 'class') { e.className = v; } else if (k === 'text') { e.textContent = v; } else if (k === 'html') { e.innerHTML = v; }
      else if (k.slice(0, 2) === 'on') { e.addEventListener(k.slice(2), v); } else { e.setAttribute(k, v === true ? '' : v); }
    });
    [].concat(children || []).forEach(function (c) { if (c !== null && c !== undefined) { e.append(c); } });
    return e;
  }
  var uid = 0;
  function nextId(p) { uid += 1; return 'pt-mem-' + p + '-' + uid; }
  var gb = Calc.fmtGb;
  /** A small (i) that shows a note on hover or focus (tap on a phone). */
  function tip(text) { return text ? el('span', {class: 'pt-mem-i', tabindex: '0', role: 'note', 'aria-label': text, 'data-tip': text, text: 'i'}) : null; }
  function shortLabel(label) { return String(label).replace(/\s*\(.*\)\s*$/, ''); }

  // ------------------------------------------------------------------------------------------------- loading
  function showLoading() {
    root.classList.add('is-loading');
    root.setAttribute('aria-busy', 'true');
    ui.loading = el('div', {class: 'pt-mem-loading', role: 'status'}, [
      el('div', {class: 'pt-mem-progress', role: 'progressbar', 'aria-label': 'Loading the calculator', 'aria-valuemin': '0',
        'aria-valuemax': String(FILES.length), 'aria-valuenow': '0'}, el('span')),
      el('p', {class: 'pt-mem-loading-step', text: 'Loading the calculator'})
    ]);
    root.querySelectorAll('.pt-mem-static').forEach(function (n) { n.hidden = true; });
    root.append(ui.loading);
  }
  function setProgress(done, label) {
    var bar = ui.loading.querySelector('.pt-mem-progress');
    bar.setAttribute('aria-valuenow', String(done));
    bar.firstChild.style.width = (100 * done / FILES.length) + '%';
    ui.loading.querySelector('.pt-mem-loading-step').textContent = 'Loading the calculator: ' + label + ' (' + done + ' of ' + FILES.length + ')';
  }
  function showError(err) {
    root.classList.remove('is-loading');
    root.removeAttribute('aria-busy');
    if (ui.loading) { ui.loading.remove(); }
    var fileHint = location.protocol === 'file:' ? ' Serve the built site (python -m http.server); a page opened as a file cannot read other files.' : '';
    root.append(el('p', {class: 'pt-mem-error', role: 'alert', text: 'The calculator could not load: ' + String(err && err.message ? err.message : err) + '.' + fileHint
      + ' The tables below and plan_memory() give the same estimates.'}));
  }
  function load() {
    var done = 0;
    return Promise.all(FILES.map(function (f) {
      return fetch(base + f[0]).then(function (r) {
        if (!r.ok) { throw new Error(f[0] + ' (HTTP ' + r.status + ')'); }
        return r.json();
      }).then(function (j) { done += 1; setProgress(done, f[1]); return j; });
    }));
  }

  // ------------------------------------------------------------------------------------------- the state
  function webglRenderer() {
    try {
      var c = document.createElement('canvas');
      var gl = c.getContext('webgl') || c.getContext('experimental-webgl');
      if (!gl) { return null; }
      var ext = gl.getExtension('WEBGL_debug_renderer_info');
      var r = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
      var lose = gl.getExtension('WEBGL_lose_context');
      if (lose) { lose.loseContext(); }
      return r ? String(r) : null;
    } catch (e) { return null; }
  }

  function initialState() {
    var renderer = webglRenderer(), gpu = Calc.matchGpu(renderer, data.gpus);
    var mem = Calc.gpuMemoryChoices(gpu);
    var cuda = Calc.cudaCapable(gpu) && mem.length > 0;
    var s = {
      machine: {ram_gb: 32, os: Calc.detectOs(navigator) || 'windows', device: cuda ? 'cuda' : 'cpu', gpu_gb: cuda ? mem[0] : 0, ram_plan_gb: null, gpu_plan_gb: null},
      gpuName: gpu && !Calc.isStandIn(gpu) ? gpu.name : '', detected: {renderer: renderer, gpu: gpu},
      modality: 'PET', model: 'pet_sinogram', user: {}, keep: {}, preset: '', fromLink: false, linkIgnored: [], dirty: false, open: {}
    };
    resetKeep(s);
    // a link to a setup (?ram=64&os=linux&gpu=24&model=pet_listmode&...): what it gives replaces the defaults
    var link = null;
    try { link = Calc.decodeState(M, location.search, data.ref); } catch (e) { link = null; }
    if (link) {
      Object.keys(link.machine).forEach(function (k) { s.machine[k] = link.machine[k]; });
      if (link.machine.os === 'macos' && !('device' in link.machine)) { s.machine.device = 'cpu'; s.machine.gpu_gb = 0; }
      if (link.gpuName !== null) { s.gpuName = link.gpuName; } else if ('gpu_gb' in link.machine) { s.gpuName = ''; }   // the link's GPU, not this browser's
      if (link.model) {
        s.model = link.model; s.modality = Calc.modelInfo(M, link.model).modality;
        s.user = link.user; s.preset = link.preset || '';
        resetKeep(s);
        Object.keys(link.keep).forEach(function (k) { s.keep[k] = link.keep[k]; });
        link.auto.forEach(function (k) { delete s.keep[k]; });
      }
      s.fromLink = true; s.linkIgnored = link.ignored;
    }
    return s;
  }

  function resetKeep(s) {
    var info = Calc.modelInfo(M, s.model), params = Calc.paramsFor(M, s.model, s.user, null);
    s.keep = {};
    info.settings.forEach(function (st) {
      if (KEEP_BY_DEFAULT.indexOf(st.name) >= 0 && params[st.name] !== null && params[st.name] !== undefined) { s.keep[st.name] = params[st.name]; }
    });
  }

  /** This setup as a link: the page's address with the inputs as its query (memory-calc.js encodeState). */
  function linkUrl() { return location.origin + location.pathname + '?' + Calc.encodeState(M, state) + '#pt-mem'; }

  function gpuByName(name) {
    if (!name) { return null; }
    var n = name.trim().toLowerCase();
    for (var i = 0; i < data.gpus.length; i++) { if (data.gpus[i].name.toLowerCase() === n) { return data.gpus[i]; } }
    return Calc.matchGpu(name, data.gpus);
  }

  // ------------------------------------------------------------------------------------------- the layout
  function build() {
    root.classList.remove('is-loading');
    root.removeAttribute('aria-busy');
    ui.loading.remove();
    ui.machine = el('div', {class: 'pt-mem-sec'});
    ui.recon = el('div', {class: 'pt-mem-sec'});
    ui.settings = el('div', {class: 'pt-mem-sec'});
    ui.adv = el('details', {class: 'pt-mem-adv'});
    ui.adv.addEventListener('toggle', function () { state.open.adv = ui.adv.open; });
    ui.inputs = el('div', {class: 'pt-mem-in'}, [ui.machine, ui.recon, ui.settings, ui.adv]);
    ui.results = el('div', {class: 'pt-mem-out', 'aria-live': 'polite'});
    root.append(el('div', {class: 'pt-mem-grid'}, [ui.inputs, ui.results]));
    ui.mini = el('button', {type: 'button', class: 'pt-mem-mini', hidden: true, onclick: function () { ui.results.scrollIntoView({behavior: 'smooth', block: 'start'}); }});
    root.append(ui.mini);
    drawMachine();
    drawRecon();
    drawResults();          // drawn with the rest: the calculator appears complete
    if ('IntersectionObserver' in window) {     // phones: a status line while the inputs are on screen and the results are not
      var seen = {inputs: false, results: false}, io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { seen[en.target === ui.inputs ? 'inputs' : 'results'] = en.isIntersecting; });
        ui.mini.hidden = !(seen.inputs && !seen.results);
      });
      io.observe(ui.inputs); io.observe(ui.results);
    }
  }

  /** A cell: a label (with an optional note) and its controls, on one line. */
  function cell(label, controls, opts) {
    opts = opts || {};
    var id = opts.id || null;
    return el('div', {class: 'pt-mem-cell' + (opts.wide ? ' is-wide' : '')}, [
      el(id ? 'label' : 'span', {class: 'pt-mem-l', for: id}, [label, tip(opts.tip)]),
      el('div', {class: 'pt-mem-c'}, controls)
    ]);
  }

  function segmented(name, options, value, onPick) {
    var g = el('div', {class: 'pt-mem-seg', role: 'group', 'aria-label': name});
    options.forEach(function (o) {
      g.append(el('button', {type: 'button', 'aria-pressed': String(o[0] === value), text: o[1], onclick: function () { onPick(o[0]); }}));
    });
    return g;
  }

  function sizeSelect(id, sizes, value, extra) {
    var sel = el('select', {id: id});
    (extra || []).forEach(function (o) { sel.append(new Option(o[1], o[0])); });
    sizes.forEach(function (n) { sel.append(new Option(n >= 1024 && n % 1024 === 0 ? (n / 1024) + ' TB' : n + ' GB', String(n))); });
    sel.append(new Option('Other…', 'other'));
    var known = sizes.indexOf(+value) >= 0 || (extra || []).some(function (o) { return o[0] === String(value); });
    sel.value = known ? String(value) : 'other';
    return sel;
  }

  function numberInput(value, attrs) {
    return el('input', Object.assign({type: 'number', step: 'any', inputmode: 'decimal', value: value === null || value === undefined ? '' : String(value)}, attrs || {}));
  }

  // ------------------------------------------------------------------------------------------- the machine
  function drawMachine() {
    var m = state.machine, f = ui.machine;
    f.textContent = '';

    var ramId = nextId('ram'), ramSel = sizeSelect(ramId, Calc.RAM_SIZES, m.ram_gb);
    var ramOther = numberInput(m.ram_gb, {min: '1', class: 'pt-mem-other', 'aria-label': 'RAM in GB'});
    ramOther.hidden = ramSel.value !== 'other';
    ramSel.addEventListener('change', function () {
      ramOther.hidden = ramSel.value !== 'other';
      if (ramSel.value !== 'other') { m.ram_gb = +ramSel.value; m.ram_plan_gb = null; drawAdvanced(); update(); } else { ramOther.focus(); }
    });
    ramOther.addEventListener('input', function () { if (+ramOther.value > 0) { m.ram_gb = +ramOther.value; m.ram_plan_gb = null; drawAdvanced(); update(); } });

    var osTip = m.os === 'windows' ? 'On Windows, GPU memory counts as RAM too.' : m.os === 'macos' ? 'No CUDA on a Mac: PyTomography runs on the CPU.' : null;
    var os = segmented('Operating system', [['windows', 'Windows'], ['linux', 'Linux'], ['macos', 'macOS']], m.os, function (v) {
      m.os = v; if (v === 'macos') { m.device = 'cpu'; m.gpu_gb = 0; } drawMachine(); update();
    });

    var listId = nextId('gpus'), nameId = nextId('gpu'), datalist = el('datalist', {id: listId});
    data.gpus.forEach(function (g) { if (!Calc.isStandIn(g)) { datalist.append(el('option', {value: g.name})); } });
    var nameInput = el('input', {id: nameId, type: 'search', list: listId, autocomplete: 'off', spellcheck: 'false', placeholder: 'GPU, e.g. RTX 4090', value: state.gpuName, 'aria-label': 'GPU'});
    nameInput.addEventListener('change', function () { pickGpuByName(nameInput.value); });
    var memSel = sizeSelect(nextId('gpumem'), Calc.GPU_SIZES, m.device === 'cuda' ? m.gpu_gb : 'none', [['none', 'No GPU (CPU)']]);
    memSel.setAttribute('aria-label', 'GPU memory');
    if (m.device !== 'cuda') { memSel.value = 'none'; }
    var memOther = numberInput(m.gpu_gb || '', {min: '0.5', class: 'pt-mem-other', 'aria-label': 'GPU memory in GB'});
    memOther.hidden = memSel.value !== 'other';
    memSel.addEventListener('change', function () {
      memOther.hidden = memSel.value !== 'other';
      if (memSel.value === 'none') { m.device = 'cpu'; m.gpu_gb = 0; }
      else if (memSel.value !== 'other') { m.device = 'cuda'; m.gpu_gb = +memSel.value; }
      else { memOther.focus(); return; }
      m.gpu_plan_gb = null; drawAdvanced(); update();
    });
    memOther.addEventListener('input', function () { if (+memOther.value > 0) { m.device = 'cuda'; m.gpu_gb = +memOther.value; m.gpu_plan_gb = null; drawAdvanced(); update(); } });

    f.append(el('div', {class: 'pt-mem-cells'}, [
      cell('RAM', [ramSel, ramOther], {id: ramId, tip: 'Browsers do not report RAM: pick yours. Free Colab has about 12 GB.'}),
      cell('System', [os], {tip: osTip}),
      cell('GPU', [nameInput, datalist, memSel, memOther], {id: nameId, wide: true})
    ]), gpuLine());
  }

  /** One line under the GPU: what the browser said, or which memory size this GPU has. */
  function gpuLine() {
    var d = state.detected, g = gpuByName(state.gpuName), m = state.machine, line = el('div', {class: 'pt-mem-note'});
    var choices = Calc.gpuMemoryChoices(g);
    if (g && choices.length > 1) {
      line.append(g.name + ': ');
      choices.forEach(function (n) {
        line.append(el('button', {type: 'button', class: 'pt-mem-chip', 'aria-pressed': String(m.device === 'cuda' && m.gpu_gb === n), text: n + ' GB',
          onclick: function () { m.device = 'cuda'; m.gpu_gb = n; m.gpu_plan_gb = null; drawMachine(); drawAdvanced(); update(); }}), ' ');
      });
      line.append('?');
    } else if (d.gpu && Calc.isStandIn(d.gpu)) {
      line.append('The browser hides the GPU model: pick yours. ', tip(d.renderer + ': ' + d.gpu.notes + '.'));
    } else if (d.gpu) {
      line.append('Detected: ' + d.gpu.name + ' ', tip('From the browser\'s WebGL renderer. Browsers never report memory; nothing is sent anywhere.'));
    } else {
      line.append('GPU not detected: pick it or its memory. ', tip(d.renderer ? 'The browser reports ' + d.renderer + '.' : null));
    }
    if (g && !Calc.cudaCapable(g)) { line.append(' · no CUDA: planned for the CPU ', tip('PyTomography runs on NVIDIA GPUs (CUDA). Pick a memory size to plan for a GPU anyway.')); }
    return line;
  }

  function pickGpuByName(name) {
    var m = state.machine, g = gpuByName(name);
    state.gpuName = g ? g.name : name;
    if (g) {
      var mem = Calc.gpuMemoryChoices(g);
      if (Calc.cudaCapable(g) && mem.length) { m.device = 'cuda'; m.gpu_gb = mem[0]; } else { m.device = 'cpu'; m.gpu_gb = 0; }
      m.gpu_plan_gb = null;
    }
    drawMachine(); drawAdvanced(); update();
  }

  // ---------------------------------------------------------------------------------- the reconstruction
  function modelsOf(modality) { return M.models().filter(function (x) { return x.modality === modality; }); }

  function drawRecon() {
    var f = ui.recon;
    f.textContent = '';
    var info = Calc.modelInfo(M, state.model), params = Calc.paramsFor(M, state.model, state.user, null);
    var cells = [];

    var presetId = nextId('preset'), sel = el('select', {id: presetId});
    sel.append(new Option('None', ''));
    MODALITIES.forEach(function (mod) {
      var group = el('optgroup', {label: mod});
      data.ref.cases.forEach(function (c) {
        if (c.model && Calc.modelInfo(M, c.model).modality === mod) { group.append(new Option(c.tutorial + ' · ' + c.step, c.id)); }
      });
      if (group.children.length) { sel.append(group); }
    });
    sel.value = state.preset;
    sel.addEventListener('change', function () { applyPreset(sel.value); });
    ui.presetSelect = sel;
    cells.push(cell('Tutorial', [sel], {id: presetId, wide: true, tip: 'Start from a tutorial\'s reconstruction and see what it measured.'}));

    var modelId = nextId('model'), msel = el('select', {id: modelId});
    modelsOf(state.modality).forEach(function (x) { msel.append(new Option(x.title.replace(/ reconstruction/, ''), x.id)); });
    msel.value = state.model;
    msel.addEventListener('change', function () { state.model = msel.value; state.preset = ''; resetKeep(state); drawRecon(); update(); });
    cells.push(cell('Model', [segmented('Modality', MODALITIES.map(function (x) { return [x, x]; }), state.modality, function (v) {
      if (v === state.modality) { return; }
      state.modality = v; state.model = modelsOf(v)[0].id; state.user = {}; state.preset = '';
      resetKeep(state); drawRecon(); update();
    }), msel], {id: modelId, wide: true}));

    if (info.params.some(function (p) { return p.name === 'crystals_per_ring'; })) { cells.push(scannerCell(params)); }

    var byName = {}, shown = {};
    info.params.forEach(function (p) { byName[p.name] = p; });
    (LAYOUT[state.modality] || []).forEach(function (spec) {
      var names = spec[1].filter(function (n) { return byName[n] && relevant(byName[n], params); });
      if (!names.length || !byName[spec[1][0]]) { return; }
      var controls = [];
      names.forEach(function (n, i) {
        shown[n] = true;
        if (i > 0) { controls.push(el('span', {class: 'pt-mem-x', text: spec[2] || ''})); }
        controls.push(control(byName[n], params[n], names.length > 1));
      });
      if (spec[1][0] === 'tof' && params.tof) { controls.push(el('span', {class: 'pt-mem-x', text: 'bins'})); }
      if (spec[1][0] === 'events') { controls.push(ui.count = el('span', {class: 'pt-mem-x', text: Calc.fmtCount(+params.events)})); }
      cells.push(cell(spec[0], controls, {tip: names.length === 1 && byName[names[0]].help ? byName[names[0]].help : null}));
    });
    f.append(el('div', {class: 'pt-mem-cells'}, cells));
    state.more = info.params.filter(function (p) {
      return !shown[p.name] && Calc.MACHINE_PARAMS.indexOf(p.name) < 0 && info.settings.every(function (s) { return s.name !== p.name; }) && relevant(p, params);
    });
    drawSettings();
    drawAdvanced();
  }

  function relevant(p, params) {
    if (p.name === 'tof_bins' || p.name === 'num_dense_tof_bins') { return !!params.tof; }
    if (p.name === 'prior_weight') { return !!params.prior; }
    if (p.name === 'sss_step') { return params.additive === 'lazy'; }
    return true;
  }

  /** The control of one parameter: a select for options, else a number. */
  function control(p, value, small) {
    var input;
    if (p.options) {
      input = el('select', {'aria-label': p.label, title: p.help || null});
      p.options.forEach(function (o) { input.append(new Option(o.label, JSON.stringify(o.value))); });
      input.value = JSON.stringify(value === undefined ? null : value);
      input.addEventListener('change', function () { setParam(p.name, JSON.parse(input.value), true); });
      return input;
    }
    input = numberInput(value, {min: p.min, max: p.max, class: small ? 'is-small' : (p.name === 'events' ? 'is-wide' : null),
      placeholder: p.type === 'number_or_null' ? 'auto' : null, 'aria-label': p.label + (p.unit && p.unit !== 'events' ? ' (' + p.unit + ')' : '')});
    input.addEventListener('input', function () {
      var v = input.value.trim();
      if (v === '' && p.type === 'number_or_null') { input.removeAttribute('aria-invalid'); setParam(p.name, null, false); return; }
      var n = +v;
      if (v !== '' && isFinite(n) && (p.min === undefined || n >= p.min) && (p.max === undefined || n <= p.max)) { input.removeAttribute('aria-invalid'); setParam(p.name, n, false); }
      else { input.setAttribute('aria-invalid', 'true'); }
    });
    return input;
  }

  function setParam(name, v, redraw) {
    state.user[name] = v;
    state.preset = '';
    if (name === 'events' && ui.count) { ui.count.textContent = Calc.fmtCount(+v); }
    if (redraw || name === 'crystals_per_ring' || name === 'rings' || name === 'tof_bins') { drawRecon(); }
    else if (ui.presetSelect) { ui.presetSelect.value = ''; }
    update();
  }

  function scannerCell(params) {
    var id = nextId('scanner'), sel = el('select', {id: id});
    sel.append(new Option('Custom', ''));
    SCANNER_GROUPS.forEach(function (gr) {
      var group = el('optgroup', {label: gr[1]});
      data.scanners.forEach(function (s) {
        if (s.type !== gr[0]) { return; }
        var o = new Option(s.name, s.id);
        o.disabled = !Calc.scannerParams(s);
        group.append(o);
      });
      if (group.children.length) { sel.append(group); }
    });
    var match = Calc.scannerFor(data.scanners, params.crystals_per_ring, params.rings);
    sel.value = match ? match.id : '';
    sel.addEventListener('change', function () {
      var p = Calc.scannerParams(data.scanners.filter(function (x) { return x.id === sel.value; })[0]);
      if (p) { Object.keys(p).forEach(function (k) { state.user[k] = p[k]; }); state.preset = ''; drawRecon(); update(); }
    });
    var s = Calc.sinogram(params.crystals_per_ring, params.rings, params.tof ? params.tof_bins : 1);
    var note = 'Every ring pair: ' + s.angles + ' × ' + s.radial + ' × ' + s.planes.toLocaleString('en-US') + (s.bins > 1 ? ' × ' + s.bins + ' TOF bins' : '') + ', float32.';
    if (match && match.tof && params.tof && !(match.vendor_sinogram && match.vendor_sinogram.tof_bins)) { note += ' TOF bins not published: ' + params.tof_bins + ' assumed.'; }
    if (s.gb > 20) { note += ' Too large to hold: use list mode or a lazy sinogram.'; }
    return cell('Scanner', [sel, el('span', {class: 'pt-mem-x', text: gb(s.gb) + ' GB sinogram'})], {id: id, wide: true, tip: note});
  }

  function applyPreset(id) {
    var c = data.ref.cases.filter(function (x) { return x.id === id; })[0];
    if (!c) { state.preset = ''; drawRecon(); update(); return; }
    var p = Calc.presetFromCase(c);
    state.model = p.model; state.modality = Calc.modelInfo(M, p.model).modality;
    state.user = p.user; state.preset = id; state.keep = {};
    KEEP_BY_DEFAULT.forEach(function (k) { if (k in p.settings) { state.keep[k] = p.settings[k]; } });
    drawRecon();
    update();
  }

  // ------------------------------------------------------------------------------------------- settings
  function drawSettings() {
    var f = ui.settings, info = Calc.modelInfo(M, state.model);
    f.textContent = '';
    ui.settingInputs = {};
    if (!info.settings.length) { f.hidden = true; return; }
    f.hidden = false;
    var params = Calc.paramsFor(M, state.model, state.user, null), byName = {}, cells = [];
    info.params.forEach(function (p) { byName[p.name] = p; });
    info.settings.forEach(function (s) {
      var p = byName[s.name] || {name: s.name}, kept = Object.prototype.hasOwnProperty.call(state.keep, s.name), input;
      var scale = s.name === 'gpu_budget' ? 1e9 : 1;           // gpu_budget is in bytes: shown in GB
      if (p.options) {
        input = el('select', {'aria-label': s.label});
        input.append(new Option('auto', 'auto'));
        p.options.forEach(function (o) { if (o.value !== null) { input.append(new Option(o.label, JSON.stringify(o.value))); } });
        input.value = kept ? JSON.stringify(state.keep[s.name]) : 'auto';
        input.addEventListener('change', function () {
          if (input.value === 'auto') { delete state.keep[s.name]; } else { state.keep[s.name] = JSON.parse(input.value); }
          update();
        });
      } else {
        input = numberInput(kept ? state.keep[s.name] / scale : '', {min: '0', placeholder: 'auto', 'aria-label': s.label + (scale > 1 ? ' (GB)' : '')});
        input.addEventListener('input', function () {
          var v = input.value.trim(), n = +v;
          if (v === '' || v === 'auto') { input.removeAttribute('aria-invalid'); delete state.keep[s.name]; update(); }
          else if (isFinite(n) && n > 0) { input.removeAttribute('aria-invalid'); state.keep[s.name] = n * scale; update(); }
          else { input.setAttribute('aria-invalid', 'true'); }
        });
      }
      ui.settingInputs[s.name] = {input: input, scale: scale};
      cells.push(cell(shortLabel(s.label) + (scale > 1 || s.name === 'memory_budget_gb' ? ' (GB)' : ''), [input],
        {tip: 'Empty: the plan chooses the fastest value that fits. A value is kept.'}));
    });
    f.append(el('div', {class: 'pt-mem-cells'}, cells));
  }

  // ------------------------------------------------------------------------------------------- advanced
  function drawAdvanced() {
    if (!ui.adv) { return; }
    var b = Calc.budgets(state.machine), m = state.machine, f = ui.adv, params = Calc.paramsFor(M, state.model, state.user, null);
    f.textContent = '';
    f.open = !!state.open.adv;
    f.append(el('summary', {text: 'Advanced'}));
    var ramIn = numberInput(+b.ram_plan_gb.toFixed(1), {min: '0.5', class: 'is-small', 'aria-label': 'RAM to plan for, GB'});
    ramIn.addEventListener('input', function () { if (+ramIn.value > 0) { m.ram_plan_gb = +ramIn.value; update(); } });
    var plan = [ramIn, el('span', {class: 'pt-mem-x', text: 'GB RAM'})];
    if (b.device === 'cuda') {
      var gpuIn = numberInput(+b.gpu_plan_gb.toFixed(1), {min: '0.1', class: 'is-small', 'aria-label': 'GPU memory to plan for, GB'});
      gpuIn.addEventListener('input', function () { if (+gpuIn.value > 0) { m.gpu_plan_gb = +gpuIn.value; update(); } });
      plan.push(gpuIn, el('span', {class: 'pt-mem-x', text: 'GB GPU'}));
    }
    var cells = [cell('Plan for', plan, {wide: true, tip: 'Like plan_memory(): installed less ' + gb(b.ram_margin_gb) + ' GB of RAM (16 GB, or a quarter below 64 GB) and 5% of the GPU. Lower it if other jobs run.'})];
    (state.more || []).forEach(function (p) {
      cells.push(cell(shortLabel(p.label), [control(p, params[p.name], false)], {tip: p.help}));
    });
    f.append(el('div', {class: 'pt-mem-cells'}, cells));
  }

  // ---------------------------------------------------------------------------------------------- results
  function update() {
    state.dirty = true;
    clearTimeout(timer);                                       // typing a number does not recompute at every key
    timer = setTimeout(drawResults, 80);
  }

  function bar(label, used, budget, note) {
    var over = used > budget + 1e-9, pct = budget > 0 ? Math.min(100, 100 * used / budget) : 100;
    return el('div', {class: 'pt-mem-bar ' + (over ? 'is-over' : 'is-ok')}, [
      el('span', {class: 'pt-mem-bar-l'}, [label, tip(note)]),
      el('span', {class: 'pt-mem-track', role: 'img', 'aria-label': label + ': ' + gb(used) + ' of ' + gb(budget) + ' GB' + (over ? ', over' : '')},
        el('span', {style: 'width:' + pct.toFixed(1) + '%'})),
      el('span', {class: 'pt-mem-bar-n', text: gb(used) + ' / ' + gb(budget) + ' GB'})
    ]);
  }

  function disclosure(key, title, body) {
    var d = el('details', {class: 'pt-mem-more'}, [el('summary', {text: title})].concat(body));
    if (state.open[key]) { d.open = true; }
    d.addEventListener('toggle', function () { state.open[key] = d.open; });
    return d;
  }

  function fmtSet(k, v) { return Calc.fmtSetting(k, v); }

  function drawResults() {
    var b = Calc.budgets(state.machine), r;
    try {
      r = Calc.recommend(M, state.model, state.user, b, state.keep);
    } catch (err) {
      ui.results.textContent = '';
      ui.results.append(el('p', {class: 'pt-mem-error', role: 'alert', text: 'These inputs cannot be estimated: ' + err.message}));
      return;
    }
    var p = r.plan, cuda = b.device === 'cuda', out = [];

    // status, with Copy link
    var copyLink = el('button', {type: 'button', class: 'pt-mem-btn', text: 'Copy link'});
    copyLink.addEventListener('click', function () { copy(linkUrl(), copyLink, 'Copy link'); });
    out.push(el('div', {class: 'pt-mem-status ' + (p.fits ? 'is-ok' : 'is-over')}, [
      el('span', {}, [el('b', {text: p.fits ? 'Fits' : 'Does not fit'}),
        ' · ' + gb(p.ram_gb) + ' GB RAM' + (cuda ? ' · ' + gb(p.gpu_gb) + ' GB GPU' : ' · CPU') + (p.fits ? '' : ' · ' + gb(p.excess_gb) + ' GB over')]),
      copyLink
    ]));
    out.push(bar('RAM', p.ram_gb, b.ram_plan_gb, cuda && b.os === 'windows' ? 'Includes the GPU memory (Windows).' : null));
    if (cuda) { out.push(bar('GPU', p.gpu_gb, b.gpu_plan_gb)); }

    // settings, name = value
    var names = Object.keys(p.settings);
    if (names.length) {
      out.push(el('p', {class: 'pt-mem-set'}, names.map(function (k) {
        return el('code', {text: k + ' = ' + fmtSet(k, p.settings[k])});
      })));
    }
    Object.keys(ui.settingInputs || {}).forEach(function (k) {         // the chosen value, in the empty inputs
      var s = ui.settingInputs[k];
      if (!Object.prototype.hasOwnProperty.call(state.keep, k) && s.input.tagName === 'INPUT') {
        var v = p.settings[k];
        s.input.placeholder = v === null || v === undefined ? 'auto' : 'auto: ' + (typeof v === 'number' ? +(v / s.scale).toPrecision(3) : v);
      }
    });

    // one-line notes
    if (state.fromLink && !state.dirty && state.linkIgnored.length) {
      out.push(el('p', {class: 'pt-mem-note', text: 'Not used from the link: ' + state.linkIgnored.join(', ') + '.'}));
    }
    if (state.preset) {
      var c = data.ref.cases.filter(function (x) { return x.id === state.preset; })[0];
      if (c) {
        var ce = Calc.caseEstimate(M, data.ref, c), mt = c.measured.tutorial;
        out.push(el('p', {class: 'pt-mem-note'}, ['Tutorial measured ' + gb(mt.ram_gb) + ' GB RAM, ' + gb(mt.gpu_gb) + ' GB GPU ',
          tip('Whole notebook (' + data.ref.machine.gpu + ', ' + data.ref.machine.ram_gb + ' GB, Windows). This step with its settings: estimated '
            + gb(ce.ram_gb) + ' GB RAM, ' + gb(ce.gpu_gb) + ' GB GPU.' + (c.notes ? ' ' + c.notes : ''))]));
      }
    }

    // disclosures: what to change, the parts, speed
    if (r.suggestions.length) {
      var nfit = r.suggestions.filter(function (s) { return s.plan.fits; }).length;
      out.push(disclosure('sugg', (p.fits ? 'Faster' : 'What to change') + ' (' + (nfit ? nfit + ' fit' : 'none fits') + ')',
        [el('ul', {class: 'pt-mem-list'}, r.suggestions.map(function (s) {
          return el('li', {class: s.plan.fits ? 'is-ok' : 'is-over'}, [
            el('button', {type: 'button', class: 'pt-mem-apply', text: 'Apply', onclick: function () { applySuggestion(s); }}),
            el('span', {title: s.label + (s.detail ? '. ' + s.detail : '')}, [el('code', {text: s.short}),
              ' ' + (s.plan.fits ? 'fits' : gb(s.plan.excess_gb) + ' GB over') + ', ' + gb(s.plan.ram_gb) + (cuda ? ' / ' + gb(s.plan.gpu_gb) : '') + ' GB'])
          ]);
        }))]));
    } else if (!p.fits) {
      out.push(el('p', {class: 'pt-mem-note', text: 'No setting makes this fit: more memory, a smaller image or fewer events.'}));
    }
    var parts = p.estimate.parts.slice().sort(function (a, z) { return (z.ram_gb + z.gpu_gb) - (a.ram_gb + a.gpu_gb); });
    out.push(disclosure('parts', 'Parts of the peak (' + parts.length + ')', [el('table', {class: 'pt-mem-parts'}, [
      el('thead', {}, el('tr', {}, [el('th', {text: 'Part'}), el('th', {text: ''}), el('th', {class: 'pt-mem-num', text: 'RAM'}), el('th', {class: 'pt-mem-num', text: 'GPU'})])),
      el('tbody', {}, parts.map(function (x) {
        return el('tr', {}, [el('td', {text: x.name}), el('td', {class: 'pt-mem-dim', text: SCOPES[x.scope] || x.scope}),
          el('td', {class: 'pt-mem-num', text: x.ram_gb ? gb(x.ram_gb) : ''}), el('td', {class: 'pt-mem-num', text: x.gpu_gb ? gb(x.gpu_gb) : ''})]);
      }))
    ])]));
    var speed = r.notes.filter(function (n) { return n.note; }).map(function (n) { return el('li', {}, [el('code', {text: n.name + ' = ' + fmtSet(n.name, n.value)}), ' ' + n.note]); });
    speed.push(el('li', {text: 'Host-side splits cost little (helical FBP: 11.3 s, 12.5 s at a 0.06 GB budget); GPU splits cost time (4.5 times as long at an eighth of the GPU budget).'}));
    out.push(disclosure('speed', 'Speed', [el('ul', {class: 'pt-mem-list'}, speed)]));

    // Python
    var copyCode = el('button', {type: 'button', class: 'pt-mem-btn pt-mem-copy', text: 'Copy'});
    copyCode.addEventListener('click', function () { copy(r.code, copyCode, 'Copy'); });
    out.push(el('div', {class: 'pt-mem-code'}, [el('pre', {}, el('code', {text: r.code})), copyCode]));

    ui.results.textContent = '';
    out.forEach(function (n) { ui.results.append(n); });
    ui.mini.className = 'pt-mem-mini ' + (p.fits ? 'is-ok' : 'is-over');
    ui.mini.textContent = (p.fits ? 'Fits' : 'Does not fit') + ' · ' + gb(p.ram_gb) + ' GB RAM' + (cuda ? ' · ' + gb(p.gpu_gb) + ' GB GPU' : '') + '  ↓';
    if (state.fromLink || state.dirty) {      // the address follows the inputs; the page's anchor stays
      try { history.replaceState(null, '', location.pathname + '?' + Calc.encodeState(M, state) + location.hash); } catch (e) { /* file:// or a sandbox */ }
    }
  }

  function copy(text, button, label) {
    var done = function () { button.textContent = 'Copied'; setTimeout(function () { button.textContent = label; }, 1500); };
    var fail = function () { window.prompt('Copy:', text); };
    if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text).then(done, fail); } else { fail(); }
  }

  function applySuggestion(s) {
    if (s.model !== state.model) { state.model = s.model; state.modality = Calc.modelInfo(M, s.model).modality; }
    state.user = Object.assign({}, s.user);
    state.keep = Object.assign({}, s.fixed || {});
    state.preset = '';
    drawRecon();
    update();
  }

  // ---------------------------------------------------------------------------------------------- start
  showLoading();
  load().then(function (files) {
    M.load(files[0]);
    data.scanners = files[1]; data.gpus = files[2]; data.ref = files[3];
    state = initialState();
    build();
  }).catch(showError);
}());
