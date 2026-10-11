---
html_theme.sidebar_secondary.remove: true
---

# Memory and speed

<link rel="stylesheet" href="_static/memory/memory.css">

<p class="pt-mem-lede">This page answers how much RAM and GPU memory a PyTomography reconstruction needs, and which settings make it fit your machine while running as fast as possible. If you can run Python on the machine, <code>plan_memory()</code> reads what is free and is exact; this page estimates from what you tell it.</p>

<ul class="pt-mem-toc" aria-label="On this page">
<li class="pt-mem-toc-label">On this page</li>
<li><a href="#for-ai-agents">For AI agents</a></li>
<li><a href="#calculator">Calculator</a></li>
<li><a href="#reference-the-tutorials">Reference: the tutorials</a></li>
<li><a href="#pet-scanners">PET scanners</a></li>
<li><a href="#how-memory-is-controlled">How memory is controlled</a></li>
<li><a href="#formulas">Formulas</a></li>
<li><a href="#sources">Sources</a></li>
</ul>

## For AI agents

Plan memory in code before running anything large. Every system matrix, algorithm, filtered back projection and scatter estimate has `estimate_memory()` (settings in, peak RAM and GPU memory out) and `plan_memory()` (memory in, the fastest settings that fit out); both print a table and give JSON with `to_dict()`.

```python
import pytomography
pytomography.set_memory_budget(24)                  # GB of RAM PyTomography's host-side steps may use
algorithm = OSEM(likelihood)                        # any algorithm, on any system matrix
print(algorithm.estimate_memory(n_subsets=28))      # peak RAM and GPU memory with these settings
plan = algorithm.plan_memory(n_subsets=28)          # fits what is free now (or pass ram_gb=, gpu_gb=)
print(plan.to_dict())                               # settings, fits, estimate, every option tried
```

Without Python on the target machine, use the same model the calculator uses. It is plain data, and `pytomo-memory.js` evaluates it in a browser or in Node (`require('./pytomo-memory.js').load(require('./model.json'))`):

| File | What it contains |
|---|---|
| <a href="_static/memory/model.json"><code>_static/memory/model.json</code></a> | The memory model: every estimate and plan of the library as expressions, with its schema in the header. |
| <a href="_static/memory/pytomo-memory.js"><code>_static/memory/pytomo-memory.js</code></a> | Its evaluator: `models()`, `estimate(id, params, settings)`, `plan(id, params, ram_gb, gpu_gb, fixed)`, `code(...)`. |
| <a href="_static/memory/formulas.md"><code>_static/memory/formulas.md</code></a> | The same estimates written out as formulas (the [Formulas](#formulas) section below). |
| <a href="_static/memory/pet_scanners.json"><code>_static/memory/pet_scanners.json</code></a> | 26 PET scanners: geometry, PyTomography `info` dicts, TOF and sensitivity, with sources. |
| <a href="_static/memory/gpus.json"><code>_static/memory/gpus.json</code></a> | 278 GPUs with their memory, and the stand-ins Firefox reports, in the order that matches a WebGL renderer string. |
| <a href="_static/memory/reference_cases.json"><code>_static/memory/reference_cases.json</code></a> | The tutorials' reconstructions as model inputs, with their measured peaks. |

The repository's `docs/tools/check_memory_model.py` checks `model.json` against the library's own `estimate_memory()` and `plan_memory()` (every estimate within 1%, every plan the same), and the formulas below are written from it.

### Link to a setup

The calculator's inputs are in its link, so a setup can be shared, and an agent can send someone to one. The **Copy link to this setup** button gives every input of the present one. For example, a 128 GB Linux workstation with a 24 GB GPU and the Biograph Vision Quadra in list mode:

<p><a href="memory.html?ram=128&amp;os=linux&amp;gpu=24&amp;model=pet_listmode&amp;crystals_per_ring=760&amp;rings=320&amp;tof=true&amp;events=500000000&amp;n_subsets=20#calculator"><code>memory.html?ram=128&amp;os=linux&amp;gpu=24&amp;model=pet_listmode&amp;crystals_per_ring=760&amp;rings=320&amp;tof=true&amp;events=500000000&amp;n_subsets=20</code></a></p>

| Key | Value |
|---|---|
| `ram` | RAM installed, in GB. |
| `os` | `windows`, `linux` or `macos`. |
| `gpu` | GPU memory in GB, or `none` to run on the CPU. |
| `gpu_name` | The GPU's name as in `gpus.json` (shown only; optional). |
| `ram_plan`, `gpu_plan` | GB to plan for, in place of what is installed less the margins `plan_memory()` keeps (16 GB of RAM, or a quarter of it below 64 GB; 5% of the GPU). |
| `model` | A model's `id` in `model.json`, such as `pet_sinogram`, `pet_listmode`, `sss`, `ct_helical_fbp`, `ct_cone_fdk`, `spect` or `starguide`. |
| `preset` | A tutorial's case `id` in `reference_cases.json`, such as `t_CT_GEN3/fbp`. It is loaded first, and the other keys change it. |
| a parameter | Any parameter of the model, by its `name` in `model.json`: `Lx`, `crystals_per_ring`, `tof`, `events`, `algorithm`, … A number, `true` or `false`, or one of its options' `value`; `auto` for "not set". Parameters left out take the model's defaults. |
| a setting | `n_subsets`, `N_splits`, `gpu_budget`, `memory_budget_gb` or `lazy`, as the model has them: a value is kept, and `auto` lets the plan choose. Left out, `n_subsets` keeps the model's default and the others are chosen. |

Keys that are not inputs of the model, and values that are not valid, are listed under the result and not used.

## Calculator

<div class="pt-mem" id="pt-mem" data-base="_static/memory/">
<p class="pt-mem-static">The calculator needs JavaScript. Without it, the reference table and the formulas below give the same estimates, and <code>plan_memory()</code> gives exact ones on the machine itself.</p>
</div>

<script src="_static/memory/pytomo-memory.js"></script>
<script src="_static/memory/memory-calc.js"></script>
<script src="_static/memory/memory-ui.js"></script>

## Reference: the tutorials

```{memory-reference}
```

### PET scanners

```{memory-scanners}
```

## How memory is controlled

**One setting for RAM.** `pytomography.set_memory_budget(gb)` tells PyTomography how much host memory its memory-heavy steps may use, and each splits its work to fit: blocks of crystal pairs (normalisation weights and the list mode sensitivity image), chunks of list mode binning, time-of-flight sinograms built lazily one subset of angles at a time, OSEM subsets (a reconstruction whose subsets would not fit stops before it starts and names the number that would), and chunks of filtered back projection. The results do not depend on the budget, and these host-side splits cost little: the helical FBP of the CT tutorial took 11.3 s with its usual budget and 12.5 s with a budget of 0.06 GB.

**GPU splits cost time.** `gpu_budget` (filtered back projection; never more than a quarter of the free GPU memory) and `N_splits` (projector passes of the PET system matrices) split GPU work into smaller batches. The same FBP took 50.3 s, 4.5 times as long, with an eighth of its GPU budget. So a plan tries the cheap splits first: the fewest projector splits, then the fewest subsets that fit, and for filtered back projection a smaller memory budget before a smaller GPU budget.

**How close the estimates are.** They are calibrated never to be low: on 17 small reconstructions the measured peaks were at most 89% of the RAM estimate and 93% of the GPU estimate. At tutorial size the GPU estimate was within 0.91 to 1.09 of what `nvidia-smi` showed, and the RAM estimates of filtered back projection were 1.2 to 1.35 times the measured peak. Every tutorial peaks under 25 GB of RAM.

**On Windows:**

- GPU memory that PyTorch holds counts as the process's RAM (committed memory), so the RAM budget has to cover both, and the calculator adds the GPU estimate to the RAM estimate.
- PyTorch's CPU allocator keeps memory it has freed committed, which the estimates count as an overhead of the arrays.
- Python, PyTorch and CUDA take 3.7 GB before the first array (2.2 GB without CUDA), on every system.

## Formulas

```{include} _static/memory/formulas.md
:start-line: 2
:heading-offset: 1
```

### Every model in full

```{memory-formulas}
```

## Sources

```{memory-sources}
```
