/* The logic of the memory calculator on the "Memory and speed" page, kept apart from the page (memory-ui.js) so that
 * Node can test it (docs/tools/check_memory_calculator.js). It works on top of PyTomoMemory (pytomo-memory.js), which
 * evaluates model.json, and on the reference data next to it: pet_scanners.json, gpus.json and the tutorials'
 * reference cases. Nothing here touches the DOM or the network.
 *
 *   PyTomoMemoryCalc.matchGpu(renderer, gpus)          // the gpus.json entry of a WebGL renderer string, or null
 *   PyTomoMemoryCalc.budgets(machine)                  // what to plan for: RAM and GPU budgets, device, os
 *   PyTomoMemoryCalc.paramsFor(M, id, user, budgets)   // the model's parameters: defaults, the user's, the machine's
 *   PyTomoMemoryCalc.recommend(M, id, user, budgets, fixed) // plan + effective peaks + suggestions + Python
 *   PyTomoMemoryCalc.scannerParams(scanner)            // crystals_per_ring, rings, tof, tof_bins of a scanner
 *   PyTomoMemoryCalc.sinogram(C, R, tofBins)           // a PyTomography sinogram's shape and size
 *   PyTomoMemoryCalc.caseEstimate(M, ref, c)           // a tutorial's estimate with its own settings and machine
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module && module.exports) { module.exports = api; }
  if (typeof window !== 'undefined') { window.PyTomoMemoryCalc = api; }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // ------------------------------------------------------------------------------------------------ the machine
  // Sizes offered in the selectors (any other value can be typed).
  var RAM_SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];
  var GPU_SIZES = [2, 4, 6, 8, 10, 11, 12, 16, 20, 24, 32, 40, 48, 64, 80, 94, 96, 128, 141, 180, 192, 256, 288];

  // The margins pytomography.utils.memory.available_memory() keeps free: 16 GB of RAM, or a quarter of it on machines
  // with less than 64 GB; 5% of the GPU.
  var RAM_MARGIN_GB = 16, RAM_MARGIN_FRACTION = 0.25, GPU_MARGIN_FRACTION = 0.05;

  // Parameters the "Your machine" inputs set, not the reconstruction's.
  var MACHINE_PARAMS = ['device', 'os', 'gpu_free_gb', 'memory_budget_gb'];

  /** The first entry of gpus.json whose `match` is a substring of the lower-cased renderer string (the file is ordered
   * so that the first match wins, e.g. "rtx 5090 laptop" before "rtx 5090"); null if none matches. */
  function matchGpu(renderer, gpus) {
    if (!renderer || !gpus) { return null; }
    var s = String(renderer).toLowerCase();
    for (var i = 0; i < gpus.length; i++) {
      if (gpus[i].match && s.indexOf(gpus[i].match) >= 0) { return gpus[i]; }
    }
    return null;
  }

  /** The GPU's memory sizes in GB, smallest first ([] if it shares the system's RAM or is unknown). */
  function gpuMemoryChoices(gpu) {
    if (!gpu || gpu.memory_gb === null || gpu.memory_gb === undefined) { return []; }
    var list = Array.isArray(gpu.memory_gb) ? gpu.memory_gb.slice() : [gpu.memory_gb];
    return list.sort(function (a, b) { return a - b; });
  }

  /** Whether PyTomography's GPU code (CUDA) runs on this GPU: NVIDIA GPUs. */
  function cudaCapable(gpu) { return !!gpu && gpu.vendor === 'NVIDIA'; }

  /** 'windows', 'macos' or 'linux' from a navigator-like object ({userAgentData, platform, userAgent}); null if unknown. */
  function detectOs(nav) {
    if (!nav) { return null; }
    var p = ((nav.userAgentData && nav.userAgentData.platform) || '') + ' ' + (nav.platform || '') + ' ' + (nav.userAgent || '');
    p = p.toLowerCase();
    if (/mac|iphone|ipad/.test(p)) { return 'macos'; }
    if (/win/.test(p)) { return 'windows'; }
    if (/linux|x11|cros|android|bsd/.test(p)) { return 'linux'; }
    return null;
  }

  /** What to plan for, from what the user says about the machine:
   *   machine = {ram_gb, os: 'windows'|'linux'|'macos', device: 'cuda'|'cpu', gpu_gb (0 or null: no GPU),
   *              ram_plan_gb (optional override), gpu_plan_gb (optional override)}
   * RAM: installed less the library's margin; GPU: its memory less 5%. On Windows the RAM budget covers the GPU memory
   * too (the estimate's RAM includes it). Without a CUDA GPU, PyTomography runs on the CPU. */
  function budgets(machine) {
    var ram = Math.max(0, +machine.ram_gb || 0);
    var gpu = Math.max(0, +machine.gpu_gb || 0);
    var device = machine.device === 'cuda' && gpu > 0 ? 'cuda' : 'cpu';
    var ramMargin = Math.min(RAM_MARGIN_GB, RAM_MARGIN_FRACTION * ram);
    var ramPlan = isNum(machine.ram_plan_gb) ? +machine.ram_plan_gb : Math.max(0, ram - ramMargin);
    var gpuPlan = device === 'cuda' ? (isNum(machine.gpu_plan_gb) ? +machine.gpu_plan_gb : gpu * (1 - GPU_MARGIN_FRACTION)) : 0;
    var os = machine.os === 'windows' ? 'windows' : 'linux';      // the model's os: macOS behaves as Linux
    return {
      ram_gb: ram, ram_margin_gb: ramMargin, ram_plan_gb: ramPlan,
      gpu_gb: device === 'cuda' ? gpu : 0, gpu_plan_gb: gpuPlan,
      device: device, os: os, os_name: machine.os || os,
      // pytomography.set_memory_budget: the RAM budget, in whole GB (at least 1)
      memory_budget_gb: Math.max(1, Math.floor(ramPlan)),
      gpu_free_gb: device === 'cuda' ? gpu : 0
    };
  }

  function isNum(x) { return x !== null && x !== undefined && x !== '' && isFinite(+x); }

  // ------------------------------------------------------------------------------------------- the model's inputs
  /** The model's description from PyTomoMemory.models(). */
  function modelInfo(M, id) {
    var all = M.models();
    for (var i = 0; i < all.length; i++) { if (all[i].id === id) { return all[i]; } }
    throw new Error('no model ' + id);
  }

  function settingNames(info) { return info.settings.map(function (s) { return s.name; }); }

  /** The model's parameters: its defaults, then the user's values (for the parameters it has), then the machine's. */
  function paramsFor(M, id, user, b) {
    var info = modelInfo(M, id), params = {};
    info.params.forEach(function (p) { params[p.name] = p['default'] === undefined ? null : p['default']; });
    Object.keys(user || {}).forEach(function (k) {
      if (Object.prototype.hasOwnProperty.call(params, k) && MACHINE_PARAMS.indexOf(k) < 0) { params[k] = user[k]; }
    });
    if (b) {
      var machine = {device: b.device, os: b.os, gpu_free_gb: b.device === 'cuda' ? b.gpu_free_gb : 0, memory_budget_gb: b.memory_budget_gb};
      Object.keys(machine).forEach(function (k) { if (Object.prototype.hasOwnProperty.call(params, k)) { params[k] = machine[k]; } });
    }
    return params;
  }

  /** Peak RAM and GPU memory of an estimate as the machine sees them. With CUDA: as estimated (on Windows the RAM
   * already includes the GPU memory). On the CPU, the parts the model puts on the GPU are in RAM too. */
  function effective(e, b) {
    if (b.device === 'cuda') { return {ram_gb: e.ram_gb, gpu_gb: e.gpu_gb}; }
    return {ram_gb: e.ram_gb + (b.os === 'windows' ? 0 : e.gpu_gb), gpu_gb: 0};
  }

  function excess(eff, b) {
    return Math.max(0, eff.ram_gb - b.ram_plan_gb) + (b.device === 'cuda' ? Math.max(0, eff.gpu_gb - b.gpu_plan_gb) : 0);
  }

  /** The search of PyTomoMemory.plan (model.json's "plan": every combination of the candidates, the first setting the
   * outer loop; the first that fits, else the least excess), with a fit test of our own: used on the CPU, where the
   * GPU parts count as RAM. With the CUDA test it gives the same settings as PyTomoMemory.plan. */
  function searchPlan(M, id, params, fixed, b) {
    var cands = M.candidates(id, params), names = Object.keys(cands);
    var present = M.estimate(id, params, {}).settings;
    var lists = names.map(function (n) {
      if (fixed && fixed[n] !== null && fixed[n] !== undefined) {
        var forcedValue = M.estimate(id, params, oneKey(n, fixed[n])).settings[n];
        return [forcedValue];                     // the value it keeps, or the one a 'force' imposes (MLEM: 1 subset)
      }
      return cands[n].length ? cands[n] : [present[n]];
    });
    var idx = names.map(function () { return 0; });
    var total = lists.reduce(function (a, l) { return a * l.length; }, 1);
    var options = [], chosen = null, closest = null, least = Infinity;
    for (var count = 0; count < total; count++) {
      var trial = {};
      names.forEach(function (n, i) { trial[n] = lists[i][idx[i]]; });
      var e = M.estimate(id, params, trial), eff = effective(e, b), ex = excess(eff, b);
      var fits = ex <= 1e-9;
      options.push({settings: e.settings, ram_gb: eff.ram_gb, gpu_gb: eff.gpu_gb, fits: fits});
      if (fits) { chosen = trial; break; }
      if (ex < least) { closest = trial; least = ex; }
      for (var j = lists.length - 1; j >= 0; j--) { idx[j]++; if (idx[j] < lists[j].length) { break; } idx[j] = 0; }
    }
    var settings = chosen !== null ? chosen : (closest || {});
    var est = M.estimate(id, params, settings);
    return {settings: est.settings, estimate: est, fits: chosen !== null, options: options};
  }

  function oneKey(k, v) { var o = {}; o[k] = v; return o; }

  /** The plan for this machine: PyTomoMemory.plan with a CUDA GPU, our own search on the CPU. */
  function planFor(M, id, params, b, fixed) {
    var p;
    if (b.device === 'cuda') {
      var keep = {}, names = settingNames(modelInfo(M, id)), cands = M.candidates(id, params);
      Object.keys(fixed || {}).forEach(function (k) {
        if (names.indexOf(k) >= 0 && k in cands && fixed[k] !== null && fixed[k] !== undefined) { keep[k] = fixed[k]; }
      });
      var r = M.plan(id, params, b.ram_plan_gb, b.gpu_plan_gb, keep);
      p = {settings: r.settings, estimate: r.estimate, fits: r.fits, options: r.options};
    } else {
      p = searchPlan(M, id, params, fixed, b);
    }
    var eff = effective(p.estimate, b);
    p.ram_gb = eff.ram_gb; p.gpu_gb = eff.gpu_gb; p.excess_gb = excess(eff, b);
    return p;
  }

  // ------------------------------------------------------------------------------------------------ suggestions
  // Inputs that change memory but not the image, and the values worth trying, most frugal first.
  var LEVERS = {
    data: {values: ['lazy', 'dense_ram', 'dense_gpu']},
    additive: {values: ['lazy', 'dense'], unless: 'none'},
    proj_device: {values: ['cpu', 'device']},
    lor_device: {values: ['cpu', 'device']},
    lm_weights: {values: ['ram', 'gpu'], unless: 'none'},
    ct_data: {values: ['ram', 'gpu']},
    cache_probabilities: {values: [false, true]},
    cache_kernel: {values: [false, true]},
    kem_kernel_on_gpu: {values: [false, true]},
    kem_support_on: {values: ['ram', 'device']},
    cupy: {values: [true]}
  };

  // Models that answer the same question with less memory (same scanner, image and events).
  var SWITCHES = {
    pet_sinogram: [{to: 'pet_listmode', why: 'List mode keeps the events instead of a sinogram of every ring pair.'}]
  };

  function optionLabel(param, v) {
    var opts = param.options || [];
    for (var i = 0; i < opts.length; i++) { if (opts[i].value === v) { return opts[i].label; } }
    return String(v);
  }

  /** Changes that make the reconstruction fit (or come closest), when the plan does not fit, or when it fits only
   * with the GPU work split: other placements of the data (LEVERS), another model (SWITCHES), and letting the plan
   * choose a setting that was kept. Each: {kind, label, detail, model, user, fixed, plan}, those that fit first,
   * then by excess, then by RAM. When no single change fits, pairs of changes that fit come first (a second change
   * on top of each of the three closest). */
  function suggestions(M, id, user, b, fixed, base) {
    var singles = singleChanges(M, id, user, b, fixed, base);
    if (!singles.length || singles[0].plan.fits) { return singles; }
    var pairs = [], seen = {};
    singles.slice(0, 3).forEach(function (s) {
      singleChanges(M, s.model, s.user, b, s.fixed, s.plan).forEach(function (t) {
        if (!t.plan.fits || (s.kind === 'model' && t.kind === 'model')) { return; }
        var label = s.label + ', and ' + t.label.charAt(0).toLowerCase() + t.label.slice(1);
        var key = stateKey(t.model, t.user, t.fixed);       // the same two changes in either order
        if (seen[key]) { return; }
        seen[key] = true;
        pairs.push({kind: 'pair', label: label, detail: s.detail || t.detail, model: t.model, user: t.user, fixed: t.fixed, plan: t.plan});
      });
    });
    return sortChanges(pairs).concat(singles);
  }

  function stateKey(model, user, fixed) {
    function sorted(o) { var r = {}; Object.keys(o || {}).sort().forEach(function (k) { r[k] = o[k]; }); return r; }
    return JSON.stringify([model, sorted(user), sorted(fixed)]);
  }

  function sortChanges(list) {
    return list.sort(function (a, z) {
      if (a.plan.fits !== z.plan.fits) { return a.plan.fits ? -1 : 1; }
      if (Math.abs(a.plan.excess_gb - z.plan.excess_gb) > 1e-9) { return a.plan.excess_gb - z.plan.excess_gb; }
      return a.plan.ram_gb - z.plan.ram_gb;
    });
  }

  function singleChanges(M, id, user, b, fixed, base) {
    var info = modelInfo(M, id), out = [];
    var params = paramsFor(M, id, user, b);
    info.params.forEach(function (p) {
      var lever = LEVERS[p.name];
      if (!lever || !p.options) { return; }
      var now = params[p.name];
      if (lever.unless !== undefined && now === lever.unless) { return; }
      lever.values.forEach(function (v) {
        if (v === now) { return; }
        if (!p.options.some(function (o) { return o.value === v; })) { return; }
        var u = Object.assign({}, user, oneKey(p.name, v));
        out.push({kind: 'lever', label: p.label + ': ' + optionLabel(p, v), name: p.name, value: v,
          model: id, user: u, fixed: fixed, plan: planFor(M, id, paramsFor(M, id, u, b), b, fixed)});
      });
    });
    (SWITCHES[id] || []).forEach(function (s) {
      var target = modelInfo(M, s.to), f = {};
      Object.keys(fixed || {}).forEach(function (k) { if (settingNames(target).indexOf(k) >= 0) { f[k] = fixed[k]; } });
      out.push({kind: 'model', label: 'Use ' + target.title, detail: s.why, model: s.to, user: user, fixed: f,
        plan: planFor(M, s.to, paramsFor(M, s.to, user, b), b, f)});
    });
    Object.keys(fixed || {}).forEach(function (k) {
      if (fixed[k] === null || fixed[k] === undefined) { return; }
      var f = Object.assign({}, fixed); delete f[k];
      var label = (info.settings.filter(function (s) { return s.name === k; })[0] || {label: k}).label;
      out.push({kind: 'unfix', label: 'Let the plan choose ' + label, name: k, model: id, user: user, fixed: f,
        plan: planFor(M, id, params, b, f)});
    });
    var baseExcess = base ? base.excess_gb : Infinity;
    out = out.filter(function (s) {
      return s.plan.fits || s.plan.excess_gb < baseExcess - 1e-6;
    });
    return sortChanges(out);
  }

  // ---------------------------------------------------------------------------------------------- speed notes
  /** One line on what each chosen setting costs in time. */
  function settingNote(name, value, info) {
    switch (name) {
      case 'N_splits':
        return value > 1 ? 'The projector runs in ' + value + ' GPU passes per subset. GPU splits cost time; more GPU memory lets it use fewer.'
                         : 'The projector runs in one pass per subset: the fastest.';
      case 'n_subsets':
        return 'Subsets change the update as well as the memory: OSEM with more subsets converges in fewer iterations. Keep the number your protocol uses if you have one.';
      case 'gpu_budget':
        return 'Bytes the filtered back projection may hold on the GPU at once. Smaller GPU batches cost time (up to 4.5 times in the helical FBP tutorial at an eighth of the budget).';
      case 'memory_budget_gb':
        return 'Host-side work is split to fit this; that costs little (helical FBP: 11.3 s to 12.5 s at 0.06 GB).';
      case 'lazy':
        return value ? 'The scatter estimate is kept lazy (interpolated per subset) rather than as a dense TOF sinogram.' : 'The scatter estimate is kept as a dense sinogram.';
      default:
        return '';
    }
  }

  // ------------------------------------------------------------------------------------------------ recommend
  /** Everything the result panel shows for one configuration. */
  function recommend(M, id, user, b, fixed) {
    var params = paramsFor(M, id, user, b);
    var plan = planFor(M, id, params, b, fixed);
    var slow = plan.fits && plan.settings.N_splits > 1;
    var sugg = (!plan.fits || slow) ? suggestions(M, id, user, b, fixed, plan) : [];
    if (slow) {      // when it fits already, only changes that fit with fewer GPU splits are worth showing
      sugg = sugg.filter(function (s) { return s.plan.fits && (s.plan.settings.N_splits || 1) < plan.settings.N_splits; });
    }
    var codeSettings = Object.assign({}, plan.settings);
    return {
      model: id, params: params, budgets: b, plan: plan, suggestions: sugg.slice(0, 4),
      code: M.code(id, params, codeSettings),
      notes: Object.keys(plan.settings).map(function (k) { return {name: k, value: plan.settings[k], note: settingNote(k, plan.settings[k])}; })
    };
  }

  // --------------------------------------------------------------------------------------------- PET scanners
  /** A PyTomography sinogram of a scanner with C crystals per ring and R rings: every ring pair, angles C/2, radial
   * bins C + 1, planes R^2, float32; times the TOF bins. */
  function sinogram(C, R, tofBins) {
    var angles = Math.floor(C / 2), radial = C + 1, planes = R * R, bins = tofBins || 1;
    var perBin = 4 * angles * radial * planes / 1e9;
    return {angles: angles, radial: radial, planes: planes, bins: bins, gb_per_bin: perBin, gb: perBin * bins};
  }

  /** The model parameters a scanner preset sets; tof_bins from the vendor's sinogram when it is published. */
  function scannerParams(s) {
    if (!s || !s.crystals_per_ring || !s.rings) { return null; }
    var p = {crystals_per_ring: s.crystals_per_ring, rings: s.rings, tof: !!s.tof};
    var vb = s.vendor_sinogram && s.vendor_sinogram.tof_bins;
    if (s.tof && vb) { p.tof_bins = vb; }
    return p;
  }

  /** The scanner of pet_scanners.json with these crystals and rings, or null. */
  function scannerFor(scanners, C, R) {
    for (var i = 0; i < (scanners || []).length; i++) {
      if (scanners[i].crystals_per_ring === C && scanners[i].rings === R) { return scanners[i]; }
    }
    return null;
  }

  // ------------------------------------------------------------------------------------------- tutorial cases
  /** A tutorial's reconstruction as the calculator's inputs: its model, the user-facing parameters and its settings. */
  function presetFromCase(c) {
    var user = {};
    Object.keys(c.params || {}).forEach(function (k) { if (MACHINE_PARAMS.indexOf(k) < 0) { user[k] = c.params[k]; } });
    return {model: c.model, user: user, settings: Object.assign({}, c.settings || {})};
  }

  /** The estimate of a tutorial's step with its own settings on the machine it was measured on (the reference file's
   * defaults: Windows, CUDA, the GPU memory free then). */
  function caseEstimate(M, ref, c) {
    var params = Object.assign({}, ref.defaults || {}, c.params || {});
    var known = {};
    modelInfo(M, c.model).params.forEach(function (p) { known[p.name] = true; });
    Object.keys(params).forEach(function (k) { if (!known[k]) { delete params[k]; } });
    return M.estimate(c.model, params, c.settings || {});
  }

  // -------------------------------------------------------------------------------------------------- display
  function fmtGb(x) {
    if (x === null || x === undefined || !isFinite(x)) { return '–'; }
    if (x >= 100) { return Math.round(x).toLocaleString('en-US'); }
    if (x >= 1) { return x.toFixed(1); }
    return x.toFixed(2);
  }

  function fmtCount(n) {
    if (!isFinite(n)) { return String(n); }
    if (n >= 1e9) { return +(n / 1e9).toPrecision(3) + ' billion'; }
    if (n >= 1e6) { return +(n / 1e6).toPrecision(3) + ' million'; }
    return Math.round(n).toLocaleString('en-US');
  }

  function fmtSetting(name, v) {
    if (v === null || v === undefined) { return name === 'memory_budget_gb' ? 'none' : 'auto'; }
    if (typeof v === 'boolean') { return v ? 'yes' : 'no'; }
    if (name === 'gpu_budget') { return fmtGb(v / 1e9) + ' GB'; }
    if (name === 'memory_budget_gb') { return String(+(+v).toFixed(2)) + ' GB'; }
    return String(v);
  }

  return {
    RAM_SIZES: RAM_SIZES, GPU_SIZES: GPU_SIZES, MACHINE_PARAMS: MACHINE_PARAMS, LEVERS: LEVERS,
    RAM_MARGIN_GB: RAM_MARGIN_GB, RAM_MARGIN_FRACTION: RAM_MARGIN_FRACTION, GPU_MARGIN_FRACTION: GPU_MARGIN_FRACTION,
    matchGpu: matchGpu, gpuMemoryChoices: gpuMemoryChoices, cudaCapable: cudaCapable, detectOs: detectOs,
    budgets: budgets, modelInfo: modelInfo, paramsFor: paramsFor, effective: effective, searchPlan: searchPlan,
    planFor: planFor, suggestions: suggestions, recommend: recommend, settingNote: settingNote,
    sinogram: sinogram, scannerParams: scannerParams, scannerFor: scannerFor,
    presetFromCase: presetFromCase, caseEstimate: caseEstimate,
    fmtGb: fmtGb, fmtCount: fmtCount, fmtSetting: fmtSetting
  };
}));
