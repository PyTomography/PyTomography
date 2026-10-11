# Memory and speed

<link rel="stylesheet" href="_static/memory/memory.css">

<p class="pt-mem-lede">This page answers how much RAM and GPU memory a PyTomography reconstruction needs, and which settings make it fit your machine while running as fast as possible. If you can run Python on the machine, <code>plan_memory()</code> reads what is free and is exact; this page estimates from what you tell it.</p>

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
| <a href="_static/memory/pet_scanners.json"><code>_static/memory/pet_scanners.json</code></a> | 26 PET scanners: geometry, PyTomography `info` dicts, TOF and sensitivity, with sources. |
| <a href="_static/memory/gpus.json"><code>_static/memory/gpus.json</code></a> | 278 GPUs with their memory, in the order that matches a WebGL renderer string. |
| <a href="_static/memory/reference_cases.json"><code>_static/memory/reference_cases.json</code></a> | The tutorials' reconstructions as model inputs, with their measured peaks. |

The [formulas](#formulas) below are generated from the same `model.json`, which the repository's `docs/tools/check_memory_model.py` checks against the library's own `estimate_memory()` and `plan_memory()` (within 1%).

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

Sizes are in bytes; images and projections are float32 (4 bytes). With an image of $L_x \times L_y \times L_z$ voxels, $C$ crystals per ring, $R$ rings, $T$ TOF bins (1 without TOF), $M$ subsets and $E$ events:

- an image takes $4 L_x L_y L_z$;
- a PET sinogram (every ring pair) takes $4 \lfloor C/2 \rfloor (C+1) R^2 T$, and a reconstruction holds three of one subset's share at a time (data, expected counts and their ratio): $3 \cdot 4 \lceil \lfloor C/2 \rfloor / M \rceil (C+1) R^2 T$;
- the PET projector works on chunks of lines of response, $88 + 8T$ bytes each, in at least $N_\text{splits}$ passes and in chunks no bigger than the GPU budget, $\min(1.5\ \text{GB},\ \tfrac14\ \text{free GPU memory})$;
- list mode keeps $E \cdot (2\ \text{or}\ 3) \cdot b$ for the events' detector IDs ($b$ = 4 or 8 bytes; 3 with TOF), $4E$ more to sort them, and $4 \cdot \tfrac{CR(CR-1)}{2}$ for a sensitivity weight per crystal pair (118 GB for the Biograph Vision Quadra);
- the peak is the sum of every part, plus 3.7 GB for Python, PyTorch and CUDA (2.2 GB on the CPU), plus PyTorch's GPU cache ($0.05\ \text{GB} + 0.2 \times$ the GPU arrays); on Windows the GPU memory is added to the RAM, with the memory the allocators keep (a fraction of the arrays).

Every model in full, as the calculator evaluates it:

```{memory-formulas}
```

## Sources

```{memory-sources}
```
