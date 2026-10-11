/* The memory calculator on the "Memory and speed" page: the page's inputs and results. The arithmetic is in
 * pytomo-memory.js (the memory model) and memory-calc.js (budgets, plans, suggestions); this file only reads inputs and
 * draws. It loads its four data files from _static/memory/ (no other network access), shows a loading bar held at the
 * calculator's size until all have arrived, then draws the calculator whole. */
(function () {
  'use strict';
  var root = document.getElementById('pt-mem');
  if (!root) { return; }
  var M = window.PyTomoMemory, Calc = window.PyTomoMemoryCalc;
  var base = root.getAttribute('data-base') || '_static/memory/';
  var FILES = [
    ['model.json', 'memory model'], ['pet_scanners.json', 'PET scanners'],
    ['gpus.json', 'GPUs'], ['reference_cases.json', 'tutorial cases']
  ];
  var MODALITIES = ['PET', 'CT', 'SPECT'];
  // Inputs shown up front; the model's other parameters go under "More options".
  var PRIMARY = {
    PET: ['crystals_per_ring', 'rings', 'tof', 'tof_bins', 'events', 'Lx', 'Ly', 'Lz', 'algorithm', 'prior', 'data', 'additive', 'sss_input'],
    CT: ['Lx', 'Ly', 'Lz', 'views', 'det_cols', 'det_rows', 'ct_data', 'views_per_rotation'],
    SPECT: ['Lx', 'Ly', 'Lz', 'views', 'proj_r', 'proj_z', 'attenuation', 'psf', 'psf_kernel', 'n_radii', 'algorithm', 'prior',
            'sg_angles', 'sg_views_per_angle', 'sg_views', 'kem_size', 'n_beds']
  };
  var SCANNER_GROUPS = [['PET/CT', 'Clinical PET/CT'], ['PET/MR', 'PET/MR'], ['total-body', 'Long axial field of view'],
                        ['brain', 'Brain'], ['preclinical', 'Preclinical']];
  var KEEP_BY_DEFAULT = ['n_subsets'];

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
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  var uid = 0;
  function nextId(p) { uid += 1; return 'pt-mem-' + p + '-' + uid; }
  var gb = Calc.fmtGb;

  // ------------------------------------------------------------------------------------------------- loading
  function showLoading() {
    root.classList.add('is-loading');
    root.setAttribute('aria-busy', 'true');
    ui.loading = el('div', {class: 'pt-mem-loading', role: 'status'}, [
      el('p', {class: 'pt-mem-loading-title', text: 'Loading the calculator'}),
      el('div', {class: 'pt-mem-progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': String(FILES.length), 'aria-valuenow': '0'},
        el('span', {class: 'pt-mem-progress-fill'})),
      el('p', {class: 'pt-mem-loading-step', text: 'The memory model, PET scanners, GPUs and tutorial cases (about 230 kB)'})
    ]);
    root.querySelectorAll('.pt-mem-static').forEach(function (n) { n.hidden = true; });
    root.append(ui.loading);
  }
  function setProgress(done, label) {
    var bar = ui.loading.querySelector('.pt-mem-progress');
    bar.setAttribute('aria-valuenow', String(done));
    bar.firstChild.style.width = (100 * done / FILES.length) + '%';
    ui.loading.querySelector('.pt-mem-loading-step').textContent = done + ' of ' + FILES.length + ' loaded: ' + label;
  }
  function showError(err) {
    root.classList.remove('is-loading');
    root.removeAttribute('aria-busy');
    if (ui.loading) { ui.loading.remove(); }
    var fileHint = location.protocol === 'file:' ? ' This page was opened as a file, and browsers do not let a file read other files: serve the built site instead (for example python -m http.server in docs/build/html).' : '';
    root.append(el('div', {class: 'pt-mem-error', role: 'alert'}, [
      el('p', {}, el('strong', {text: 'The calculator could not load.'})),
      el('p', {text: String(err && err.message ? err.message : err) + '.' + fileHint}),
      el('p', {text: 'The rest of this page does not need it: the reference table, the formulas and plan_memory() give the same numbers.'})
    ]));
  }
  function load() {
    var done = 0;
    return Promise.all(FILES.map(function (f) {
      return fetch(base + f[0]).then(function (r) {
        if (!r.ok) { throw new Error('Could not load ' + f[0] + ' (HTTP ' + r.status + ')'); }
        return r.json();
      }).then(function (j) { done += 1; setProgress(done, f[1]); return j; });
    }));
  }

  // ---------------------------------------------------------------------------------------- machine detection
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
    var os = Calc.detectOs(navigator) || 'windows';
    var mem = Calc.gpuMemoryChoices(gpu);
    var cuda = Calc.cudaCapable(gpu) && mem.length > 0;
    var s = {
      machine: {ram_gb: 32, os: os, device: cuda ? 'cuda' : 'cpu', gpu_gb: cuda ? mem[0] : 0, ram_plan_gb: null, gpu_plan_gb: null},
      gpuName: gpu ? gpu.name : '', detected: {renderer: renderer, gpu: gpu, os: Calc.detectOs(navigator)},
      modality: 'PET', model: 'pet_sinogram', user: {}, keep: {}, preset: ''
    };
    resetKeep(s);
    return s;
  }

  function resetKeep(s) {
    var info = Calc.modelInfo(M, s.model), params = Calc.paramsFor(M, s.model, s.user, null);
    s.keep = {};
    info.settings.forEach(function (st) {
      if (KEEP_BY_DEFAULT.indexOf(st.name) >= 0 && params[st.name] !== null && params[st.name] !== undefined) { s.keep[st.name] = params[st.name]; }
    });
  }

  // ------------------------------------------------------------------------------------------- the layout
  function build() {
    root.classList.remove('is-loading');
    root.removeAttribute('aria-busy');
    ui.loading.remove();
    ui.inputs = el('div', {class: 'pt-mem-inputs'});
    ui.results = el('div', {class: 'pt-mem-results', 'aria-live': 'polite'});
    ui.machine = el('fieldset', {class: 'pt-mem-step'});
    ui.recon = el('fieldset', {class: 'pt-mem-step'});
    ui.settings = el('fieldset', {class: 'pt-mem-step'});
    ui.inputs.append(ui.machine, ui.recon, ui.settings);
    root.append(el('div', {class: 'pt-mem-grid'}, [ui.inputs, ui.results]));
    ui.mini = el('button', {type: 'button', class: 'pt-mem-mini', hidden: true, onclick: function () { ui.results.scrollIntoView({behavior: 'smooth', block: 'start'}); }});
    root.append(ui.mini);
    if ('IntersectionObserver' in window) {     // shown while the inputs are on screen and the results are not
      var seen = {inputs: false, results: false};
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { seen[en.target === ui.inputs ? 'inputs' : 'results'] = en.isIntersecting; });
        ui.mini.hidden = !(seen.inputs && !seen.results);
      });
      io.observe(ui.inputs); io.observe(ui.results);
    }
    drawMachine();
    drawRecon();
    drawResults();          // the first results are drawn with the rest: the calculator appears complete
  }

  function legend(n, text) { return el('legend', {}, [el('span', {class: 'pt-mem-n', text: n}), text]); }

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
    sizes.forEach(function (n) { sel.append(new Option(n >= 1024 && n % 1024 === 0 ? (n / 1024) + ' TB (' + n + ' GB)' : n + ' GB', String(n))); });
    sel.append(new Option('Other…', 'other'));
    var known = sizes.indexOf(+value) >= 0 || (extra || []).some(function (o) { return o[0] === String(value); });
    sel.value = known ? String(value) : 'other';
    return sel;
  }

  // ------------------------------------------------------------------------------------------- 1 · machine
  function drawMachine() {
    var m = state.machine, f = ui.machine;
    f.textContent = '';
    f.append(legend('1', 'Your machine'));

    // RAM
    var ramId = nextId('ram'), ramSel = sizeSelect(ramId, Calc.RAM_SIZES, m.ram_gb);
    var ramOther = el('input', {type: 'number', min: '1', max: '100000', step: 'any', inputmode: 'decimal', class: 'pt-mem-other', 'aria-label': 'RAM in GB', value: String(m.ram_gb)});
    ramOther.hidden = ramSel.value !== 'other';
    ramSel.addEventListener('change', function () {
      ramOther.hidden = ramSel.value !== 'other';
      if (ramSel.value !== 'other') { m.ram_gb = +ramSel.value; m.ram_plan_gb = null; drawBudgets(); update(); } else { ramOther.focus(); }
    });
    ramOther.addEventListener('input', function () { if (+ramOther.value > 0) { m.ram_gb = +ramOther.value; m.ram_plan_gb = null; drawBudgets(); update(); } });
    f.append(el('div', {class: 'pt-mem-row'}, [
      el('label', {for: ramId, class: 'pt-mem-label', text: 'RAM'}),
      el('div', {class: 'pt-mem-control'}, [ramSel, ramOther]),
      el('p', {class: 'pt-mem-hint', text: 'Browsers do not tell web pages how much memory a computer has, so pick yours. Colab\'s free tier has about 12 GB.'})
    ]));

    // operating system
    var osNames = [['windows', 'Windows'], ['linux', 'Linux'], ['macos', 'macOS']];
    f.append(el('div', {class: 'pt-mem-row'}, [
      el('span', {class: 'pt-mem-label', text: 'Operating system'}),
      segmented('Operating system', osNames, m.os, function (v) { m.os = v; if (v === 'macos') { m.device = 'cpu'; } drawMachine(); update(); }),
      el('p', {class: 'pt-mem-hint', text: m.os === 'windows' ? 'On Windows, GPU memory PyTorch holds counts as RAM too, so the RAM estimate includes it.'
        : m.os === 'macos' ? 'On a Mac PyTomography runs on the CPU (it uses CUDA for the GPU), so everything is in RAM.' : 'On Linux, GPU memory is separate from RAM.'})
    ]));

    // GPU: name (searchable), memory
    var listId = nextId('gpus'), nameId = nextId('gpu');
    var datalist = el('datalist', {id: listId});
    data.gpus.forEach(function (g) { datalist.append(el('option', {value: g.name})); });
    var nameInput = el('input', {id: nameId, type: 'search', list: listId, autocomplete: 'off', spellcheck: 'false', placeholder: 'Search, e.g. RTX 4090, A100, M3', value: state.gpuName});
    nameInput.addEventListener('change', function () { pickGpuByName(nameInput.value); });
    var memId = nextId('gpumem');
    var memSel = sizeSelect(memId, Calc.GPU_SIZES, m.device === 'cuda' ? m.gpu_gb : 'none', [['none', 'No GPU, or not NVIDIA (CPU only)']]);
    if (m.device !== 'cuda') { memSel.value = 'none'; }
    var memOther = el('input', {type: 'number', min: '0.5', max: '4096', step: 'any', inputmode: 'decimal', class: 'pt-mem-other', 'aria-label': 'GPU memory in GB', value: String(m.gpu_gb || '')});
    memOther.hidden = memSel.value !== 'other';
    memSel.addEventListener('change', function () {
      memOther.hidden = memSel.value !== 'other';
      if (memSel.value === 'none') { m.device = 'cpu'; m.gpu_gb = 0; }
      else if (memSel.value !== 'other') { m.device = 'cuda'; m.gpu_gb = +memSel.value; }
      else { memOther.focus(); return; }
      m.gpu_plan_gb = null; drawBudgets(); update();
    });
    memOther.addEventListener('input', function () { if (+memOther.value > 0) { m.device = 'cuda'; m.gpu_gb = +memOther.value; m.gpu_plan_gb = null; drawBudgets(); update(); } });

    var variants = el('div', {class: 'pt-mem-variants'});
    var g = gpuByName(state.gpuName), choices = Calc.gpuMemoryChoices(g);
    if (g && choices.length > 1) {
      variants.append(el('span', {text: g.name + ' comes with '}));
      choices.forEach(function (n, i) {
        variants.append(el('button', {type: 'button', class: 'pt-mem-chip', 'aria-pressed': String(m.device === 'cuda' && m.gpu_gb === n), text: n + ' GB',
          onclick: function () { m.device = 'cuda'; m.gpu_gb = n; m.gpu_plan_gb = null; drawMachine(); update(); }}));
        if (i < choices.length - 1) { variants.append(document.createTextNode(' ')); }
      });
      variants.append(el('span', {text: ': which is yours?'}));
    }

    f.append(el('div', {class: 'pt-mem-row'}, [
      el('label', {for: nameId, class: 'pt-mem-label', text: 'GPU'}),
      el('div', {class: 'pt-mem-control pt-mem-gpu'}, [nameInput, datalist]),
      el('p', {class: 'pt-mem-hint pt-mem-detected', html: detectedText()}),
      variants.childNodes.length ? variants : null
    ]));
    f.append(el('div', {class: 'pt-mem-row'}, [
      el('label', {for: memId, class: 'pt-mem-label', text: 'GPU memory'}),
      el('div', {class: 'pt-mem-control'}, [memSel, memOther])
    ]));

    ui.budgets = el('div', {class: 'pt-mem-budgets'});
    f.append(ui.budgets);
    drawBudgets();
  }

  function gpuByName(name) {
    if (!name) { return null; }
    var n = name.trim().toLowerCase();
    for (var i = 0; i < data.gpus.length; i++) { if (data.gpus[i].name.toLowerCase() === n) { return data.gpus[i]; } }
    return Calc.matchGpu(name, data.gpus);
  }

  function pickGpuByName(name) {
    var m = state.machine, g = gpuByName(name);
    state.gpuName = g ? g.name : name;
    if (g) {
      var mem = Calc.gpuMemoryChoices(g);
      if (Calc.cudaCapable(g) && mem.length) { m.device = 'cuda'; m.gpu_gb = mem[0]; }
      else { m.device = 'cpu'; m.gpu_gb = 0; }
      m.gpu_plan_gb = null;
    }
    drawMachine();
    update();
  }

  function detectedText() {
    var d = state.detected, g = gpuByName(state.gpuName);
    var parts = [];
    if (d.gpu) {
      parts.push('Detected from your browser: <b>' + esc(d.gpu.name) + '</b>.');
    } else if (d.renderer) {
      parts.push('Your browser reports <code>' + esc(d.renderer) + '</code>, which is not in the list: pick your GPU or its memory.');
    } else {
      parts.push('Your browser does not say which GPU this is: pick it or its memory.');
    }
    if (g) {
      var mem = Calc.gpuMemoryChoices(g);
      if (!Calc.cudaCapable(g)) {
        parts.push(g.kind === 'unified' ? 'It shares the system\'s RAM and has no CUDA, so the plan is for the CPU.'
          : 'PyTomography\'s GPU code uses CUDA (NVIDIA), so the plan is for the CPU; choose a memory size to plan for the GPU anyway (e.g. with ROCm).');
      } else if (mem.length === 0) {
        parts.push('Its memory is shared with the system: pick how much to plan for.');
      }
      if (g.notes) { parts.push('<span class="pt-mem-muted">(' + esc(g.notes) + ')</span>'); }
    }
    parts.push('Browsers report a GPU\'s name at most, never its memory; nothing is sent anywhere.');
    return parts.join(' ');
  }

  function drawBudgets() {
    var b = Calc.budgets(state.machine), m = state.machine, box = ui.budgets;
    box.textContent = '';
    var ramIn = el('input', {type: 'number', min: '0.5', step: 'any', inputmode: 'decimal', value: String(+b.ram_plan_gb.toFixed(1))});
    ramIn.addEventListener('input', function () { if (+ramIn.value > 0) { m.ram_plan_gb = +ramIn.value; update(); } });
    var row = [el('label', {}, ['RAM to plan for', el('span', {class: 'pt-mem-unit'}, [ramIn, 'GB'])])];
    if (b.device === 'cuda') {
      var gpuIn = el('input', {type: 'number', min: '0.1', step: 'any', inputmode: 'decimal', value: String(+b.gpu_plan_gb.toFixed(1))});
      gpuIn.addEventListener('input', function () { if (+gpuIn.value > 0) { m.gpu_plan_gb = +gpuIn.value; update(); } });
      row.push(el('label', {}, ['GPU memory to plan for', el('span', {class: 'pt-mem-unit'}, [gpuIn, 'GB'])]));
    } else {
      row.push(el('label', {}, ['GPU', el('span', {class: 'pt-mem-unit', text: 'none: on the CPU'})]));
    }
    box.append(el('div', {class: 'pt-mem-plan-for'}, row));
    box.append(el('p', {class: 'pt-mem-hint', text: 'Like plan_memory(), this keeps ' + gb(b.ram_margin_gb) + ' GB of RAM free for everything else (16 GB, or a quarter of the RAM below 64 GB)'
      + (b.device === 'cuda' ? ' and 5% of the GPU' : '') + '. Lower it if other programs or jobs are running.'}));
  }

  // ------------------------------------------------------------------------------------- 2 · reconstruction
  function modelsOf(modality) { return M.models().filter(function (x) { return x.modality === modality; }); }

  function drawRecon() {
    var f = ui.recon;
    f.textContent = '';
    f.append(legend('2', 'Your reconstruction'));

    // tutorial presets
    var presetId = nextId('preset'), sel = el('select', {id: presetId});
    sel.append(new Option('Your own (no tutorial)', ''));
    MODALITIES.forEach(function (mod) {
      var group = el('optgroup', {label: mod});
      data.ref.cases.forEach(function (c) {
        if (!c.model || Calc.modelInfo(M, c.model).modality !== mod) { return; }
        group.append(new Option(c.tutorial + ' · ' + c.step, c.id));
      });
      if (group.children.length) { sel.append(group); }
    });
    sel.value = state.preset;
    sel.addEventListener('change', function () { applyPreset(sel.value); });
    f.append(el('div', {class: 'pt-mem-row'}, [
      el('label', {for: presetId, class: 'pt-mem-label', text: 'Start from a tutorial'}),
      el('div', {class: 'pt-mem-control'}, sel),
      el('p', {class: 'pt-mem-hint', text: 'Loads a tutorial\'s reconstruction (image, data, settings) and shows what it measured.'})
    ]));

    // modality and model
    f.append(el('div', {class: 'pt-mem-row'}, [
      el('span', {class: 'pt-mem-label', text: 'Modality'}),
      segmented('Modality', MODALITIES.map(function (x) { return [x, x]; }), state.modality, function (v) {
        if (v === state.modality) { return; }
        state.modality = v; state.model = modelsOf(v)[0].id; state.user = {}; state.preset = '';
        resetKeep(state); drawRecon(); update();
      })
    ]));
    var modelId = nextId('model'), msel = el('select', {id: modelId});
    modelsOf(state.modality).forEach(function (x) { msel.append(new Option(x.title, x.id)); });
    msel.value = state.model;
    msel.addEventListener('change', function () { state.model = msel.value; state.preset = ''; resetKeep(state); drawRecon(); update(); });
    f.append(el('div', {class: 'pt-mem-row'}, [
      el('label', {for: modelId, class: 'pt-mem-label', text: 'Reconstruction'}),
      el('div', {class: 'pt-mem-control'}, msel)
    ]));

    var info = Calc.modelInfo(M, state.model), params = Calc.paramsFor(M, state.model, state.user, null);
    if (info.params.some(function (p) { return p.name === 'crystals_per_ring'; })) { f.append(drawScanner(params)); updateScannerLine(); }

    // parameters
    var settingNames = info.settings.map(function (s) { return s.name; });
    var shown = info.params.filter(function (p) { return Calc.MACHINE_PARAMS.indexOf(p.name) < 0 && settingNames.indexOf(p.name) < 0 && relevant(p, params); });
    var primary = PRIMARY[state.modality] || [];
    var main = shown.filter(function (p) { return primary.indexOf(p.name) >= 0; });
    var more = shown.filter(function (p) { return primary.indexOf(p.name) < 0; });
    f.append(el('div', {class: 'pt-mem-fields'}, main.map(function (p) { return field(p, params[p.name], false); })));
    if (more.length) {
      var det = el('details', {class: 'pt-mem-more'}, [el('summary', {text: 'More options (' + more.length + ')'}),
        el('div', {class: 'pt-mem-fields'}, more.map(function (p) { return field(p, params[p.name], true); }))]);
      if (state.moreOpen) { det.open = true; }
      det.addEventListener('toggle', function () { state.moreOpen = det.open; });
      f.append(det);
    }
    drawSettings();
  }

  function relevant(p, params) {
    if (p.name === 'tof_bins' || p.name === 'num_dense_tof_bins') { return !!params.tof; }
    if (p.name === 'prior_weight') { return !!params.prior; }
    if (p.name === 'sss_step') { return params.additive === 'lazy'; }
    return true;
  }

  function field(p, value, withHelp) {
    var id = nextId(p.name), input;
    var unit = p.unit && p.unit !== 'events' ? ' (' + p.unit + ')' : '';
    if (p.options) {
      input = el('select', {id: id});
      p.options.forEach(function (o) { input.append(new Option(o.label, JSON.stringify(o.value))); });
      input.value = JSON.stringify(value === undefined ? null : value);
      input.addEventListener('change', function () { setParam(p.name, JSON.parse(input.value), true); });
    } else {
      input = el('input', {id: id, type: 'number', step: 'any', inputmode: 'decimal', min: p.min, max: p.max,
        placeholder: p.type === 'number_or_null' ? 'auto' : null, value: value === null || value === undefined ? '' : String(value)});
      input.addEventListener('input', function () {
        var v = input.value.trim();
        if (v === '' && p.type === 'number_or_null') { input.removeAttribute('aria-invalid'); setParam(p.name, null, false); return; }
        var n = +v;
        var ok = v !== '' && isFinite(n) && (p.min === undefined || n >= p.min) && (p.max === undefined || n <= p.max);
        if (ok) { input.removeAttribute('aria-invalid'); setParam(p.name, n, false); } else { input.setAttribute('aria-invalid', 'true'); }
      });
    }
    var kids = [el('label', {for: id, class: 'pt-mem-label', text: p.label + unit}), input];
    if (p.name === 'events') { kids.push(el('span', {class: 'pt-mem-hint pt-mem-count', text: Calc.fmtCount(+value)})); }
    if (withHelp && p.help) { kids.push(el('span', {class: 'pt-mem-hint', text: p.help})); }
    var w = el('div', {class: 'pt-mem-field', title: p.help || null}, kids);
    return w;
  }

  function setParam(name, v, redraw) {
    state.user[name] = v;
    state.preset = '';
    if (name === 'events') {
      var c = ui.recon.querySelector('.pt-mem-count');
      if (c) { c.textContent = Calc.fmtCount(+v); }
    }
    if (name === 'crystals_per_ring' || name === 'rings' || name === 'tof' || name === 'tof_bins') { updateScannerLine(); }
    if (redraw) { drawRecon(); } else { clearPresetSelect(); }
    update();
  }

  function clearPresetSelect() {
    var sel = ui.recon.querySelector('select');
    if (sel) { sel.value = ''; }
  }

  // PET scanner presets
  function drawScanner(params) {
    var id = nextId('scanner'), sel = el('select', {id: id});
    sel.append(new Option('Custom geometry', ''));
    SCANNER_GROUPS.forEach(function (gr) {
      var group = el('optgroup', {label: gr[1]});
      data.scanners.forEach(function (s) {
        if (s.type !== gr[0]) { return; }
        var ok = Calc.scannerParams(s);
        var o = new Option(s.name + (ok ? ' · ' + s.crystals_per_ring + ' × ' + s.rings : ' (geometry not published)'), s.id);
        o.disabled = !ok;
        group.append(o);
      });
      if (group.children.length) { sel.append(group); }
    });
    var match = Calc.scannerFor(data.scanners, params.crystals_per_ring, params.rings);
    sel.value = match ? match.id : '';
    sel.addEventListener('change', function () {
      var s = data.scanners.filter(function (x) { return x.id === sel.value; })[0];
      var p = Calc.scannerParams(s);
      if (p) { Object.keys(p).forEach(function (k) { state.user[k] = p[k]; }); state.preset = ''; drawRecon(); update(); }
    });
    ui.scannerLine = el('p', {class: 'pt-mem-hint pt-mem-sino'});
    var row = el('div', {class: 'pt-mem-row'}, [
      el('label', {for: id, class: 'pt-mem-label', text: 'PET scanner'}),
      el('div', {class: 'pt-mem-control'}, sel),
      ui.scannerLine
    ]);
    return row;
  }

  function updateScannerLine() {
    var p = Calc.paramsFor(M, state.model, state.user, null);
    if (!ui.scannerLine || !p.crystals_per_ring || !p.rings) { return; }
    var s = Calc.sinogram(p.crystals_per_ring, p.rings, p.tof ? p.tof_bins : 1);
    var scanner = Calc.scannerFor(data.scanners, p.crystals_per_ring, p.rings);
    var text = 'A PyTomography sinogram keeps every ring pair: ' + s.angles + ' angles × ' + s.radial + ' radial bins × '
      + s.planes.toLocaleString('en-US') + ' planes' + (s.bins > 1 ? ' × ' + s.bins + ' TOF bins' : '') + ' = '
      + gb(s.gb) + ' GB' + (s.bins > 1 ? ' (' + gb(s.gb_per_bin) + ' GB without TOF)' : '') + '.';
    if (scanner && scanner.tof && p.tof && !(scanner.vendor_sinogram && scanner.vendor_sinogram.tof_bins)) {
      text += ' The scanner\'s TOF bins are not published; ' + p.tof_bins + ' is assumed.';
    }
    if (s.gb > 20) { text += ' That is large: list mode, or a lazy sinogram binned from the events, avoids holding it.'; }
    ui.scannerLine.textContent = text;
  }

  function applyPreset(id) {
    var c = data.ref.cases.filter(function (x) { return x.id === id; })[0];
    if (!c) { state.preset = ''; drawRecon(); update(); return; }
    var p = Calc.presetFromCase(c);
    state.model = p.model; state.modality = Calc.modelInfo(M, p.model).modality;
    state.user = p.user; state.preset = id;
    state.keep = {};
    KEEP_BY_DEFAULT.forEach(function (k) { if (k in p.settings) { state.keep[k] = p.settings[k]; } });
    drawRecon();
    update();
  }

  // ----------------------------------------------------------------------------------------- 3 · settings
  function drawSettings() {
    var f = ui.settings, info = Calc.modelInfo(M, state.model);
    f.textContent = '';
    f.append(legend('3', 'Settings'));
    if (!info.settings.length) {
      f.append(el('p', {class: 'pt-mem-hint', text: 'This reconstruction has no memory settings: its peak is fixed by the inputs above.'}));
      return;
    }
    f.append(el('p', {class: 'pt-mem-hint', text: 'Let the plan choose each setting (the fastest that fits), or keep your own value.'}));
    var params = Calc.paramsFor(M, state.model, state.user, null);
    info.settings.forEach(function (s) {
      var kept = Object.prototype.hasOwnProperty.call(state.keep, s.name);
      var id = nextId('set');
      var mode = el('select', {id: id, 'aria-label': s.label + ': choose or keep'});
      mode.append(new Option('Choose for me', 'choose'), new Option('Keep', 'keep'));
      mode.value = kept ? 'keep' : 'choose';
      var current = kept ? state.keep[s.name] : params[s.name];
      var val = el('input', {type: 'number', step: 'any', inputmode: 'decimal', 'aria-label': s.label + ' value',
        value: current === null || current === undefined ? '' : String(current), placeholder: 'value'});
      val.disabled = !kept;
      mode.addEventListener('change', function () {
        if (mode.value === 'keep') {
          val.disabled = false;
          var v = val.value === '' ? null : +val.value;
          if (v === null || !isFinite(v)) { v = params[s.name]; val.value = v === null || v === undefined ? '' : String(v); }
          state.keep[s.name] = v;
        } else { val.disabled = true; delete state.keep[s.name]; }
        update();
      });
      val.addEventListener('input', function () {
        var n = +val.value;
        if (val.value !== '' && isFinite(n) && n > 0) { val.removeAttribute('aria-invalid'); state.keep[s.name] = n; update(); }
        else { val.setAttribute('aria-invalid', 'true'); }
      });
      f.append(el('div', {class: 'pt-mem-setting'}, [
        el('label', {for: id, class: 'pt-mem-label', text: s.label}), mode, val
      ]));
    });
  }

  // ---------------------------------------------------------------------------------------------- results
  function update() {
    // a short debounce: typing a number does not recompute at every key (a large plan takes a few hundred ms)
    clearTimeout(timer);
    timer = setTimeout(drawResults, 80);
  }

  function bar(label, used, budget, total, note) {
    var over = used > budget + 1e-9;
    var pct = budget > 0 ? Math.min(100, 100 * used / budget) : 100;
    return el('div', {class: 'pt-mem-bar' + (over ? ' is-over' : ' is-ok')}, [
      el('div', {class: 'pt-mem-bar-head'}, [
        el('span', {class: 'pt-mem-bar-label', text: label}),
        el('span', {class: 'pt-mem-bar-num', html: '<b>' + gb(used) + ' GB</b> of ' + gb(budget) + ' GB' + (total ? ' <span class="pt-mem-muted">(' + gb(total) + ' GB installed)</span>' : '')})
      ]),
      el('div', {class: 'pt-mem-track', role: 'img', 'aria-label': label + ': ' + gb(used) + ' GB of a ' + gb(budget) + ' GB budget' + (over ? ', over budget' : '')},
        el('span', {class: 'pt-mem-fill', style: 'width:' + pct.toFixed(1) + '%'})),
      note ? el('p', {class: 'pt-mem-hint', text: note}) : null
    ]);
  }

  function settingsText(settings) {
    var keys = Object.keys(settings);
    if (!keys.length) { return 'no settings'; }
    return keys.map(function (k) { return k + ' ' + Calc.fmtSetting(k, settings[k]); }).join(', ');
  }

  function drawResults() {
    var b = Calc.budgets(state.machine), r;
    try {
      r = Calc.recommend(M, state.model, state.user, b, state.keep);
    } catch (err) {
      ui.results.textContent = '';
      ui.results.append(el('div', {class: 'pt-mem-error', role: 'alert', text: 'These inputs cannot be estimated: ' + err.message}));
      return;
    }
    var p = r.plan, info = Calc.modelInfo(M, state.model), out = [];

    // status
    var keptNames = Object.keys(state.keep);
    out.push(el('div', {class: 'pt-mem-status ' + (p.fits ? 'is-ok' : 'is-over')}, [
      el('p', {class: 'pt-mem-status-title', text: p.fits ? 'Fits' : 'Does not fit'}),
      el('p', {text: p.fits
        ? 'With ' + settingsText(p.settings) + ', the peak is about ' + gb(p.ram_gb) + ' GB of RAM' + (b.device === 'cuda' ? ' and ' + gb(p.gpu_gb) + ' GB of GPU memory.' : ', on the CPU.')
        : 'Closest: ' + settingsText(p.settings) + ', about ' + gb(p.ram_gb) + ' GB of RAM' + (b.device === 'cuda' ? ' and ' + gb(p.gpu_gb) + ' GB of GPU memory' : '') + ', ' + gb(p.excess_gb) + ' GB over budget.'})
    ]));

    // bars
    out.push(el('div', {class: 'pt-mem-bars'}, [
      bar('RAM', p.ram_gb, b.ram_plan_gb, b.ram_gb, b.device === 'cuda' && b.os === 'windows' ? 'On Windows this includes the GPU memory.' : (b.device === 'cpu' ? 'On the CPU, the arrays the GPU would hold are in RAM.' : null)),
      b.device === 'cuda' ? bar('GPU', p.gpu_gb, b.gpu_plan_gb, b.gpu_gb) : null
    ]));

    // settings
    if (r.notes.length) {
      var dl = el('dl', {class: 'pt-mem-settings'});
      r.notes.forEach(function (n) {
        var label = (info.settings.filter(function (s) { return s.name === n.name; })[0] || {label: n.name}).label;
        var kept = keptNames.indexOf(n.name) >= 0;
        dl.append(el('dt', {}, [label, el('span', {class: 'pt-mem-tag', text: kept ? 'kept' : 'chosen'})]),
                  el('dd', {}, [el('b', {text: Calc.fmtSetting(n.name, n.value)}), n.note ? el('span', {text: ' ' + n.note}) : null]));
      });
      out.push(el('h4', {class: 'pt-mem-h', text: p.fits ? 'Recommended settings' : 'Closest settings'}), dl);
    }

    // suggestions
    if (r.suggestions.length) {
      out.push(el('h4', {class: 'pt-mem-h', text: p.fits ? 'To run faster' : 'What to change'}));
      var ul = el('ul', {class: 'pt-mem-sugg'});
      r.suggestions.forEach(function (s) {
        ul.append(el('li', {class: s.plan.fits ? 'is-ok' : 'is-over'}, [
          el('button', {type: 'button', class: 'pt-mem-apply', text: 'Apply', 'aria-label': 'Apply: ' + s.label, onclick: function () { applySuggestion(s); }}),
          el('span', {}, [el('b', {text: s.label}), document.createTextNode(': ' + (s.plan.fits ? 'fits' : 'still ' + gb(s.plan.excess_gb) + ' GB over')
            + ', ' + gb(s.plan.ram_gb) + ' GB RAM' + (b.device === 'cuda' ? ', ' + gb(s.plan.gpu_gb) + ' GB GPU' : '') + ' with ' + settingsText(s.plan.settings) + '.'
            + (s.detail ? ' ' + s.detail : ''))])
        ]));
      });
      out.push(ul);
    } else if (!p.fits) {
      out.push(el('p', {class: 'pt-mem-hint', text: 'No setting or placement of the data makes this fit: the largest parts below set the minimum. Use a machine with more memory, a smaller image, or fewer events.'}));
    }
    if (!p.fits) {
      var biggest = p.estimate.parts.filter(function (x) { return x.scope !== 'overhead'; })
        .sort(function (a, z) { return (z.ram_gb + z.gpu_gb) - (a.ram_gb + a.gpu_gb); })[0];
      if (biggest) { out.push(el('p', {class: 'pt-mem-hint', text: 'The largest part: ' + biggest.name + ', ' + gb(biggest.ram_gb + biggest.gpu_gb) + ' GB.'})); }
    }

    // the tutorial's measurement
    if (state.preset) {
      var c = data.ref.cases.filter(function (x) { return x.id === state.preset; })[0];
      if (c) {
        var ce = Calc.caseEstimate(M, data.ref, c), mt = c.measured.tutorial, cell = c.measured.cell;
        var mach = data.ref.machine;
        out.push(el('div', {class: 'pt-mem-case'}, [
          el('h4', {class: 'pt-mem-h', text: 'The tutorial, as it ran'}),
          el('p', {text: 'With its own settings (' + settingsText(ce.settings) + ') on ' + mach.os.replace('windows', 'Windows') + ', ' + mach.gpu + ' and ' + mach.ram_gb
            + ' GB of RAM, this step is estimated at ' + gb(ce.ram_gb) + ' GB of RAM and ' + gb(ce.gpu_gb) + ' GB of GPU memory. The whole notebook peaked at '
            + gb(mt.ram_gb) + ' GB of RAM and ' + gb(mt.gpu_gb) + ' GB of GPU memory (everything it keeps, not only this step).'
            + (cell ? ' This cell alone: ' + gb(cell.ram_peak_gb - cell.ram_before_gb) + ' GB of RAM more than before it (' + gb(cell.ram_peak_gb) + ' GB in all) and ' + gb(cell.gpu_peak_torch_gb) + ' GB of GPU memory.' : '')}),
          c.notes ? el('p', {class: 'pt-mem-hint', text: c.notes}) : null
        ]));
      }
    }

    // parts
    var tbody = el('tbody');
    var scopes = {held: 'held for the whole run', subset: 'one subset at a time', chunk: 'one projector chunk at a time', overhead: 'overhead'};
    p.estimate.parts.forEach(function (part) {
      tbody.append(el('tr', {}, [el('td', {text: part.name}), el('td', {text: scopes[part.scope] || part.scope}),
        el('td', {class: 'pt-mem-num', text: part.ram_gb ? gb(part.ram_gb) : ''}), el('td', {class: 'pt-mem-num', text: part.gpu_gb ? gb(part.gpu_gb) : ''})]));
    });
    var det = el('details', {class: 'pt-mem-parts'}, [
      el('summary', {text: 'What the peak is made of (' + p.estimate.parts.length + ' parts)'}),
      el('div', {class: 'pt-mem-table-wrap'}, el('table', {}, [
        el('thead', {}, el('tr', {}, [el('th', {text: 'Part'}), el('th', {text: 'Kept'}), el('th', {class: 'pt-mem-num', text: 'RAM GB'}), el('th', {class: 'pt-mem-num', text: 'GPU GB'})])),
        tbody
      ])),
      el('p', {class: 'pt-mem-hint', text: 'Every part counts at the peak, whatever its scope.' + (b.os === 'windows' && b.device === 'cuda' ? ' The RAM total also includes the GPU column (Windows).' : '')
        + ' The estimate is never low on the tutorials: measured peaks were at most 89% of the RAM estimate and 93% of the GPU estimate.'})
    ]);
    if (state.partsOpen) { det.open = true; }
    det.addEventListener('toggle', function () { state.partsOpen = det.open; });
    out.push(det);

    // speed
    out.push(el('p', {class: 'pt-mem-speed', html: '<b>Speed.</b> Splitting host-side work costs little: the helical FBP tutorial took 11.3 s with its usual budget and 12.5 s with a 0.06 GB memory budget. Splitting GPU work costs time: 50.3 s, 4.5 times as long, with an eighth of the GPU budget. So the plan splits host-side work first and the GPU\'s last.'}));

    // code
    var codeText = '# ' + info.title + ', planned for ' + gb(b.ram_plan_gb) + ' GB RAM' + (b.device === 'cuda' ? ' and ' + gb(b.gpu_plan_gb) + ' GB GPU' : ' on the CPU') + '\n' + r.code;
    var copy = el('button', {type: 'button', class: 'pt-mem-copy', text: 'Copy'});
    copy.addEventListener('click', function () {
      var done = function () { copy.textContent = 'Copied'; setTimeout(function () { copy.textContent = 'Copy'; }, 1500); };
      if (navigator.clipboard) { navigator.clipboard.writeText(codeText).then(done, function () {}); }
    });
    out.push(el('div', {class: 'pt-mem-code'}, [
      el('div', {class: 'pt-mem-code-head'}, [el('span', {text: 'Python'}), copy]),
      el('pre', {}, el('code', {text: codeText}))
    ]));
    out.push(el('p', {class: 'pt-mem-hint', text: 'On the machine itself, algorithm.plan_memory() reads what is free and gives the exact plan.'}));

    ui.results.textContent = '';
    out.forEach(function (n) { if (n) { ui.results.append(n); } });
    var hidden = ui.mini.hidden;
    ui.mini.className = 'pt-mem-mini ' + (p.fits ? 'is-ok' : 'is-over');
    ui.mini.hidden = hidden;
    ui.mini.innerHTML = '<span><b>' + (p.fits ? 'Fits' : 'Does not fit') + '</b> \u00b7 ' + gb(p.ram_gb) + ' GB RAM'
      + (b.device === 'cuda' ? ' \u00b7 ' + gb(p.gpu_gb) + ' GB GPU' : '') + '</span><span>Results \u2193</span>';
  }

  function applySuggestion(s) {
    if (s.model !== state.model) {
      state.model = s.model;
      state.modality = Calc.modelInfo(M, s.model).modality;
    }
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
