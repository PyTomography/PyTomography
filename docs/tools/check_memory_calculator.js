#!/usr/bin/env node
/* Checks the logic of the memory calculator on the "Memory and speed" page (docs/source/_static/memory/memory-calc.js)
 * with Node, no browser and no dependencies:
 *
 *   node docs/tools/check_memory_calculator.js            (PYTHON=... to choose the Python that runs memory_model.py)
 *
 * 1. GPU detection: sample WebGL renderer strings map to the right entry of gpus.json.
 * 2. PET scanners: the sinogram sizes of pet_scanners.json in PyTomography's convention (mMR 224 x 449 x 4096, 1.65 GB).
 * 3. Tutorial presets: each tutorial's reconstruction, loaded as the calculator loads it, gives the estimate that
 *    memory_model.py (the Python evaluator the library is checked against) gives for the reference case.
 * 4. Plans: the calculator's own plan search (used on the CPU) chooses what PyTomoMemory.plan chooses with a GPU.
 * 5. Budgets, operating systems and recommendations.
 * Exits 1 if anything fails. */
'use strict';
const path = require('path');
const { execFileSync } = require('child_process');

const HERE = __dirname;
const STATIC = path.join(HERE, '..', 'source', '_static', 'memory');
const M = require(path.join(STATIC, 'pytomo-memory.js'));
const Calc = require(path.join(STATIC, 'memory-calc.js'));
M.load(require(path.join(STATIC, 'model.json')));
const gpus = require(path.join(STATIC, 'gpus.json'));
const scanners = require(path.join(STATIC, 'pet_scanners.json'));
const ref = require(path.join(HERE, 'memory_reference_cases.json'));

let failures = 0, checks = 0;
function check(ok, what) {
  checks += 1;
  if (!ok) { failures += 1; console.log('  FAIL ' + what); }
}
function close(a, b, rel) { return Math.abs(a - b) <= rel * Math.max(Math.abs(a), Math.abs(b), 1e-12); }

// ------------------------------------------------------------------------------------------------ 1. GPU detection
console.log('1. GPU detection from WebGL renderer strings');
const RENDERERS = [
  ['ANGLE (NVIDIA, NVIDIA GeForce RTX 5090 (0x00002B85) Direct3D11 vs_5_0 ps_5_0, D3D11)', 'GeForce RTX 5090', [32]],
  ['ANGLE (Apple, ANGLE Metal Renderer: Apple M3 Max, Unspecified Version)', 'Apple M3 Max', [36, 48, 64, 96, 128]],
  ['ANGLE (NVIDIA, NVIDIA GeForce RTX 5090 Laptop GPU (0x00002C58) Direct3D11 vs_5_0 ps_5_0, D3D11)', 'GeForce RTX 5090 Laptop GPU', [24]],
  ['ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Ti (0x00002803) Direct3D11 vs_5_0 ps_5_0, D3D11)', 'GeForce RTX 4060 Ti', [8, 16]],
  ['ANGLE (NVIDIA, NVIDIA GeForce RTX 4090 Direct3D11 vs_5_0 ps_5_0, D3D11)', 'GeForce RTX 4090', [24]],
  ['ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 3080/PCIe/SSE2, OpenGL 4.5.0)', 'GeForce RTX 3080', [10, 12]],
  ['ANGLE (AMD, AMD Radeon RX 7900 XTX (0x0000744C) Direct3D11 vs_5_0 ps_5_0, D3D11)', 'AMD Radeon RX 7900 XTX', [24]],
  ['ANGLE (Intel, Intel(R) UHD Graphics 620 (0x00005917) Direct3D11 vs_5_0 ps_5_0, D3D11)', 'Intel UHD Graphics', []],
  ['Apple GPU', null, []],
  ['', null, []],
];
for (const [renderer, name, mem] of RENDERERS) {
  const g = Calc.matchGpu(renderer, gpus);
  const got = g ? g.name : null;
  const gotMem = Calc.gpuMemoryChoices(g);
  const okName = name === null ? (got === null || /apple gpu/i.test(got)) : got === name;
  check(okName, `${JSON.stringify(renderer)} -> ${got} (expected ${name})`);
  if (name !== null) { check(JSON.stringify(gotMem) === JSON.stringify(mem), `${name} memory ${JSON.stringify(gotMem)} (expected ${JSON.stringify(mem)})`); }
  console.log(`  ${(renderer || '(none)').slice(0, 72).padEnd(72)} -> ${got || 'no match'}${gotMem.length ? ' (' + gotMem.join(' or ') + ' GB)' : ''}`);
}
check(Calc.cudaCapable(Calc.matchGpu(RENDERERS[0][0], gpus)) && !Calc.cudaCapable(Calc.matchGpu(RENDERERS[1][0], gpus)), 'CUDA: NVIDIA yes, Apple no');

// --------------------------------------------------------------------------------------------- 2. PET sinograms
console.log('2. PET sinogram sizes (PyTomography: C/2 angles, C + 1 radial bins, R^2 planes, float32)');
const mmr = Calc.sinogram(448, 64, 1);
check(mmr.angles === 224 && mmr.radial === 449 && mmr.planes === 4096, 'mMR shape 224 x 449 x 4096');
check(mmr.gb.toFixed(2) === '1.65', `mMR ${mmr.gb.toFixed(3)} GB (expected 1.65)`);
check(Calc.sinogram(448, 64, 21).gb.toFixed(1) === '34.6', 'mMR with 21 TOF bins: 34.6 GB (LazySinogram docstring)');
const EXPECTED = { 'siemens-biograph-mmr': 1.65, 'siemens-biograph-mct-4r': 2.11, 'siemens-biograph-vision-600': 7.40,
  'siemens-biograph-vision-quadra': 118.45, 'ge-discovery-690': 0.38, 'ge-discovery-mi-4ring': 0.77, 'ge-signa-pet-mr': 0.81,
  'united-imaging-uexplorer': 638.03, 'upenn-pennpet-explorer': 75.04, 'siemens-hrrt': 7.19, 'siemens-inveon': 1.31 };
for (const s of scanners) {
  const p = Calc.scannerParams(s);
  if (!p) { console.log(`  ${s.name.padEnd(56)} geometry not published: no preset`); continue; }
  const sino = Calc.sinogram(p.crystals_per_ring, p.rings, 1);
  // the model's own sinogram (pet_scanner derived set) agrees with the calculator's
  const e = M.estimate('pet_sinogram', { crystals_per_ring: p.crystals_per_ring, rings: p.rings, tof: false, sensitivity: true, device: 'cuda', proj_device: 'cpu', data: 'dense_ram', additive: 'none' }, { n_subsets: 1, N_splits: 1 });
  const part = e.parts.find((x) => x.name === 'sensitivity sinogram');
  check(close(part.ram_gb, sino.gb, 1e-12), `${s.name}: model's sensitivity sinogram ${part.ram_gb} GB vs ${sino.gb} GB`);
  if (s.id in EXPECTED) { check(sino.gb.toFixed(2) === EXPECTED[s.id].toFixed(2), `${s.name}: ${sino.gb.toFixed(2)} GB (sources.md: ${EXPECTED[s.id]})`); }
  console.log(`  ${s.name.padEnd(56)} ${String(sino.angles).padStart(4)} x ${String(sino.radial).padStart(4)} x ${String(sino.planes).padStart(6)} = ${sino.gb.toFixed(2).padStart(7)} GB${p.tof_bins ? `, x ${p.tof_bins} TOF bins = ${(sino.gb * p.tof_bins).toFixed(1)} GB` : ''}`);
}

// ----------------------------------------------------------------------------------------- 3. tutorial presets
console.log('3. Tutorial presets against memory_model.py');
const python = process.env.PYTHON || 'python';
const script = [
  'import json, sys',
  'import memory_model as mm',
  'm = mm.MemoryModel.load()',
  "ref = json.load(open('memory_reference_cases.json', encoding='utf8'))",
  'out = {}',
  "for c in ref['cases']:",
  "    if not c.get('model'): continue",
  "    known = {p['name'] for p in next(x for x in m.models() if x['id'] == c['model'])['params']}",
  "    p = {k: v for k, v in {**ref.get('defaults', {}), **c['params']}.items() if k in known}",
  "    e = m.estimate(c['model'], p, c.get('settings', {}))",
  "    out[c['id']] = {'ram_gb': e['ram_gb'], 'gpu_gb': e['gpu_gb'], 'settings': e['settings']}",
  'print(json.dumps(out))',
].join('\n');
let expected = null;
try {
  expected = JSON.parse(execFileSync(python, ['-c', script], { cwd: HERE, encoding: 'utf8' }));
} catch (err) {
  console.log('  FAIL could not run memory_model.py with ' + python + ': ' + err.message.split('\n')[0]);
  failures += 1;
}
if (expected) {
  for (const c of ref.cases) {
    if (!c.model) { continue; }
    const preset = Calc.presetFromCase(c);
    // the calculator's path: the preset's inputs, the machine the tutorial ran on, the tutorial's settings
    const machine = { device: 'cuda', os: ref.defaults.os, gpu_free_gb: ref.defaults.gpu_free_gb,
      memory_budget_gb: c.params.memory_budget_gb === undefined ? null : c.params.memory_budget_gb };
    const params = Calc.paramsFor(M, preset.model, preset.user, machine);
    const ui = M.estimate(preset.model, params, preset.settings);
    const direct = Calc.caseEstimate(M, ref, c);
    const py = expected[c.id];
    const ok = close(ui.ram_gb, py.ram_gb, 1e-9) && close(ui.gpu_gb, py.gpu_gb, 1e-9) && close(direct.ram_gb, py.ram_gb, 1e-9) && close(direct.gpu_gb, py.gpu_gb, 1e-9);
    check(ok, `${c.id}: calculator ${ui.ram_gb.toFixed(3)} / ${ui.gpu_gb.toFixed(3)} GB, memory_model.py ${py.ram_gb.toFixed(3)} / ${py.gpu_gb.toFixed(3)} GB`);
    const cell = c.measured.cell;
    const lib = cell && cell.library_ram_gb !== undefined ? `; library printed ${cell.library_ram_gb} / ${cell.library_gpu_gb}` : '';
    console.log(`  ${c.id.padEnd(36)} ${ui.ram_gb.toFixed(2).padStart(6)} GB RAM ${ui.gpu_gb.toFixed(2).padStart(5)} GB GPU  (python ${py.ram_gb.toFixed(2)} / ${py.gpu_gb.toFixed(2)}${lib}) ${ok ? 'ok' : 'MISMATCH'}`);
  }
}

// ------------------------------------------------------------------------------------------------- 4. plans
console.log('4. The calculator\'s plan search against PyTomoMemory.plan');
let plans = 0;
for (const info of M.models()) {
  const params = Calc.paramsFor(M, info.id, {}, { device: 'cuda', os: 'windows', gpu_free_gb: 24, memory_budget_gb: 16 });
  for (const ram of [3, 6, 12, 24, 48, 1000]) {
    for (const gpu of [0.5, 2, 8, 24]) {
      const b = { device: 'cuda', os: 'windows', ram_plan_gb: ram, gpu_plan_gb: gpu };
      const fixedSets = [{}, info.settings.some((s) => s.name === 'n_subsets') ? { n_subsets: 4 } : {}];
      for (const fixed of fixedSets) {
        const a = M.plan(info.id, params, ram, gpu, fixed);
        const s = Calc.searchPlan(M, info.id, params, fixed, b);
        plans += 1;
        check(JSON.stringify(a.settings) === JSON.stringify(s.settings) && a.fits === s.fits && a.options.length === s.options.length,
          `${info.id} at ${ram} GB / ${gpu} GB ${JSON.stringify(fixed)}: plan ${JSON.stringify(a.settings)} ${a.fits}, search ${JSON.stringify(s.settings)} ${s.fits}`);
      }
    }
  }
}
console.log(`  ${plans} plans compared over ${M.models().length} models`);

// ------------------------------------------------------------------------------------------- 5. the rest
console.log('5. Budgets, operating systems, recommendations');
let b = Calc.budgets({ ram_gb: 64, gpu_gb: 32, os: 'windows', device: 'cuda' });
check(b.ram_plan_gb === 48 && close(b.gpu_plan_gb, 30.4, 1e-12) && b.device === 'cuda' && b.os === 'windows', `64 GB / 32 GB: ${JSON.stringify(b)}`);
b = Calc.budgets({ ram_gb: 16, gpu_gb: 8, os: 'linux', device: 'cuda' });
check(b.ram_plan_gb === 12 && b.memory_budget_gb === 12, '16 GB keeps a quarter free: 12 GB');
b = Calc.budgets({ ram_gb: 32, gpu_gb: 0, os: 'macos', device: 'cuda' });
check(b.device === 'cpu' && b.os === 'linux' && b.gpu_plan_gb === 0, 'no GPU: CPU; macOS plans as Linux');
check(Calc.detectOs({ userAgentData: { platform: 'Windows' } }) === 'windows', 'detectOs Windows');
check(Calc.detectOs({ platform: 'MacIntel', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5)' }) === 'macos', 'detectOs macOS');
check(Calc.detectOs({ platform: 'Linux x86_64', userAgent: 'Mozilla/5.0 (X11; Linux x86_64)' }) === 'linux', 'detectOs Linux');

// a tutorial on the machine it ran on fits; the same on a 16 GB laptop needs changes
const tof = ref.cases.find((c) => c.id === 't_PETGATE_scat_sinoTOF/osem');
const pre = Calc.presetFromCase(tof);
let r = Calc.recommend(M, pre.model, pre.user, Calc.budgets({ ram_gb: 64, gpu_gb: 32, os: 'windows', device: 'cuda' }), { n_subsets: 28 });
check(r.plan.fits, `TOF sinogram tutorial on 64 GB / RTX 5090 fits: ${JSON.stringify(r.plan.settings)} ${r.plan.ram_gb.toFixed(1)} GB`);
console.log(`  t_PETGATE_scat_sinoTOF on 64 GB, RTX 5090:  fits=${r.plan.fits} ${JSON.stringify(r.plan.settings)} ${r.plan.ram_gb.toFixed(1)} GB RAM ${r.plan.gpu_gb.toFixed(1)} GB GPU`);
r = Calc.recommend(M, pre.model, pre.user, Calc.budgets({ ram_gb: 16, gpu_gb: 8, os: 'windows', device: 'cuda' }), { n_subsets: 28 });
console.log(`  the same on 16 GB, 8 GB GPU: fits=${r.plan.fits} ${JSON.stringify(r.plan.settings)} ${r.plan.ram_gb.toFixed(1)} GB RAM ${r.plan.gpu_gb.toFixed(1)} GB GPU`);
r.suggestions.forEach((s) => console.log(`     suggestion: ${s.label}: fits=${s.plan.fits} ${s.plan.ram_gb.toFixed(1)} / ${s.plan.gpu_gb.toFixed(1)} GB ${JSON.stringify(s.plan.settings)}`));

// a long axial FOV scanner: the sinogram cannot fit; list mode is the first suggestion
const quadra = Calc.scannerParams(scanners.find((s) => s.id === 'siemens-biograph-vision-quadra'));
r = Calc.recommend(M, 'pet_sinogram', Object.assign({ events: 500e6 }, quadra), Calc.budgets({ ram_gb: 512, gpu_gb: 80, os: 'linux', device: 'cuda' }), { n_subsets: 20 });
check(!r.plan.fits && r.suggestions.length > 0 && r.suggestions[0].model === 'pet_listmode', 'Quadra TOF sinogram: does not fit; list mode suggested first');
console.log(`  Quadra TOF sinogram on 512 GB / 80 GB: fits=${r.plan.fits}; first suggestion: ${r.suggestions[0] && r.suggestions[0].label} (fits=${r.suggestions[0] && r.suggestions[0].plan.fits}, ${r.suggestions[0] && r.suggestions[0].plan.ram_gb.toFixed(0)} GB RAM)`);

// CPU only: GPU parts are counted in RAM
const cpu = Calc.recommend(M, 'spect', {}, Calc.budgets({ ram_gb: 16, gpu_gb: 0, os: 'linux', device: 'cpu' }), { n_subsets: 8 });
const raw = M.estimate('spect', cpu.params, cpu.plan.settings);
check(close(cpu.plan.ram_gb, raw.ram_gb + raw.gpu_gb, 1e-12) && cpu.plan.gpu_gb === 0, 'CPU: RAM includes the GPU parts');

console.log(`\n${checks} checks, ${failures} failed`);
process.exit(failures ? 1 : 0);
