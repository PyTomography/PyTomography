---
html_theme.sidebar_secondary.remove: true
---

# Architecture

<div class="pt-arch-page">
<link rel="stylesheet" href="_static/arch/pt-arch.css">
<link rel="stylesheet" href="_static/arch/pt-arch-overview.css">

<div class="pt-arch-tabs" role="tablist" aria-label="Views of the library">
  <button type="button" role="tab" id="t-overview" aria-controls="tab-overview" aria-selected="true">Overview</button>
  <button type="button" role="tab" id="t-modality" aria-controls="tab-modality" aria-selected="false" tabindex="-1">By modality</button>
  <button type="button" role="tab" id="t-swap" aria-controls="tab-swap" aria-selected="false" tabindex="-1">Swap a part</button>
  <button type="button" role="tab" id="t-extend" aria-controls="tab-extend" aria-selected="false" tabindex="-1">Build your own</button>
</div>

<section class="pt-arch-panel" id="tab-overview" role="tabpanel" aria-labelledby="t-overview">
<div id="pt-arch-overview"></div>
</section>

<section class="pt-arch-panel" id="tab-modality" role="tabpanel" aria-labelledby="t-modality" hidden>
<p class="pt-arch-intro">Each modality has its own files, readers, metadata and system matrix. From the likelihood on, all three share the same classes.</p>
<div class="pt-arch-bar">
  <div class="pt-arch-seg" id="pt-arch-modality" role="group" aria-label="Show one modality"></div>
  <button type="button" class="pt-arch-btn is-primary" id="pt-arch-walk-start">Walk through one reconstruction</button>
  <p class="pt-arch-hint">Hover a box to follow its path. Click it for details.</p>
</div>
<div id="pt-arch-map" role="group" aria-label="Data, readers, standard objects and system matrix for each modality, then one likelihood, algorithm and reconstruction for all"></div>
<section id="pt-arch-walk" class="pt-arch-walk" aria-live="polite" hidden></section>
</section>

<section class="pt-arch-panel" id="tab-swap" role="tabpanel" aria-labelledby="t-swap" hidden>
<p class="pt-arch-intro">The algorithm never sees the scanner, and the system matrix never sees the statistics. Pick a change to see which parts you write and which stay as they are.</p>
<div id="pt-arch-calls" class="pt-arch-calls"></div>
</section>

<section class="pt-arch-panel" id="tab-extend" role="tabpanel" aria-labelledby="t-extend" hidden>
<p class="pt-arch-intro">Each part is a class you can subclass. Implement its few methods and every other part works with it unchanged.</p>

<div class="pt-arch-extend">

| To add | Subclass | Implement | Example |
|---|---|---|---|
| A scanner or geometry | `SystemMatrix` | `forward`, `backward`, `set_n_subsets`, `get_projection_subset`, `get_weighting_subset`, `compute_normalization_factor` | [Write your own system matrix](notebooks/t_examplesystemmatrix.ipynb) |
| A physical effect | `Transform` | `configure`, `forward`, `backward` | `SPECTPSFTransform` |
| A noise model | `Likelihood` | `compute_gradient` | `NegativeMSELikelihood` |
| A penalty | `NearestNeighbourPrior` | `phi0`, `phi1` (and `phi2_*` for uncertainty) | `RelativeDifferencePrior` |
| An update rule | `PreconditionedGradientAscentAlgorithm` | `_compute_preconditioner` | `OSEM`, `BSREM` |
| Something after every subiteration | `Callback` | `run`, `finalize` | `DataStorageCallback` |
| A file format | a reader function | return the standard objects: `object_meta`, `proj_meta`, the data as tensors | `io.SPECT.simind` |

</div>

<p class="pt-arch-intro">The mathematics behind these parts and the conventions they share (coordinates, units, the transpose rule) are in <a href="concepts.html">Concepts</a>.</p>
</section>

<script src="_static/arch/pt-arch-data.js"></script>
<script src="_static/arch/pt-arch-catalog.js"></script>
<script src="_static/arch/pt-arch.js"></script>
<script src="_static/arch/pt-arch-overview.js"></script>
</div>
