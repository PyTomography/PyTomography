---
html_theme.sidebar_secondary.remove: true
---

# Memory and speed

<link rel="stylesheet" href="_static/memory/memory.css">

<p class="pt-mem-lede">How much RAM and GPU memory a reconstruction needs, and the fastest settings that fit. On the machine itself, <code>plan_memory()</code> is exact.</p>

<div class="pt-mem" id="pt-mem" data-base="_static/memory/">
<p class="pt-mem-static">The calculator needs JavaScript. The tables below and <code>plan_memory()</code> give the same estimates.</p>
</div>

<script src="_static/memory/pytomo-memory.js"></script>
<script src="_static/memory/memory-calc.js"></script>
<script src="_static/memory/memory-ui.js"></script>

## For AI agents

Ask the library before a large run:

```python
import pytomography
pytomography.set_memory_budget(24)                  # GB of RAM for host-side work
algorithm = OSEM(likelihood)
print(algorithm.estimate_memory(n_subsets=28))      # settings in, peak RAM and GPU out
plan = algorithm.plan_memory(n_subsets=28)          # free memory in, fastest settings out
print(plan.to_dict())                               # as JSON
```

Without Python on the target machine, use the calculator's model and data, or link to a setup:

| | |
|---|---|
| <a href="_static/memory/model.json"><code>model.json</code></a>, <a href="_static/memory/pytomo-memory.js"><code>pytomo-memory.js</code></a> | The memory model and its evaluator (`models()`, `estimate()`, `plan()`, `code()`; works in Node). Checked against the library within 1%. |
| <a href="_static/memory/formulas.md"><code>formulas.md</code></a> | The same estimates as formulas. |
| <a href="_static/memory/pet_scanners.json"><code>pet_scanners.json</code></a>, <a href="_static/memory/gpus.json"><code>gpus.json</code></a>, <a href="_static/memory/reference_cases.json"><code>reference_cases.json</code></a> | 26 PET scanners, 278 GPUs, the tutorials with their measured peaks. |
| Link keys | `ram`, `os` (`windows`, `linux`, `macos`), `gpu` (GB or `none`), `gpu_name`, `ram_plan`, `gpu_plan`, `model` (an id in `model.json`), `preset` (a tutorial case), and any parameter or setting of the model by its name in `model.json`; `auto` lets the plan choose. Example: <a href="memory.html?ram=128&amp;os=linux&amp;gpu=24&amp;model=pet_listmode&amp;crystals_per_ring=760&amp;rings=320&amp;tof=true&amp;events=500000000&amp;n_subsets=20"><code>?ram=128&amp;os=linux&amp;gpu=24&amp;model=pet_listmode&amp;crystals_per_ring=760&amp;rings=320&amp;tof=true&amp;events=500000000&amp;n_subsets=20</code></a> |

## Reference

```{memory-reference}
```

## Formulas

```{memory-formulas}
```

## How memory is controlled

- `pytomography.set_memory_budget(gb)` is the one RAM setting. Host-side work splits to fit it: crystal-pair blocks, list-mode binning, lazy TOF sinograms, OSEM subsets, FBP chunks. It costs little: helical FBP took 11.3 s, and 12.5 s with a 0.06 GB budget.
- `gpu_budget` and `N_splits` split GPU work, which costs time: the same FBP took 4.5 times as long with an eighth of its GPU budget. Plans split host-side work first.
- Estimates are never low: measured peaks were at most 89% of the RAM estimate and 93% of the GPU estimate. Every tutorial peaks under 25 GB.
- On Windows, GPU memory counts as RAM, and PyTorch's allocator keeps freed memory committed. Python, PyTorch and CUDA take 3.7 GB before any array (2.2 GB without CUDA).

```{memory-sources}
```
