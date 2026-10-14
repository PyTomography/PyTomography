---
html_theme.sidebar_secondary.remove: true
---

# Architecture

<div class="pt-arch-page">
<link rel="stylesheet" href="_static/arch/pt-arch.css">
<link rel="stylesheet" href="_static/arch/pt-arch-overview.css">
<link rel="stylesheet" href="_static/arch/pt-io.css">

<div class="pt-arch-tabs" role="tablist" aria-label="Views of the library">
  <button type="button" role="tab" id="t-overview" aria-controls="tab-overview" aria-selected="true">Overview</button>
  <button type="button" role="tab" id="t-modality" aria-controls="tab-modality" aria-selected="false" tabindex="-1">By modality</button>
  <button type="button" role="tab" id="t-extend" aria-controls="tab-extend" aria-selected="false" tabindex="-1">Build your own</button>
  <button type="button" role="tab" id="t-io" aria-controls="tab-io" aria-selected="false" tabindex="-1">Data and conventions</button>
</div>

<section class="pt-arch-panel" id="tab-overview" role="tabpanel" aria-labelledby="t-overview">
<div id="pt-arch-overview"></div>

<div class="pt-arch-analytic" id="overview-analytic">
<h3>Analytic reconstruction skips the likelihood</h3>
<p class="pt-arch-intro">Filtered back projection has no likelihood, prior or subsets. <code>FilteredBackProjection(projections, system_matrix, filter=...)</code> asks the system matrix for its own geometry's inverse: it filters each view with a ramp times the window, then back projects it.</p>

| Scan | Method | The call | Tutorials |
|---|---|---|---|
| SPECT | FBP of the photopeak less scatter, then Chang's attenuation correction from `H_mu`, a system matrix with only the attenuation transform | `FilteredBackProjection(photopeak - scatter, system_matrix, filter=butterworth)()`, then `* proj_meta.num_projections / H_mu.compute_normalization_factor()` | [Filtered back projection (SPECT)](notebooks/t_fbp_spect.ipynb) |
| Helical CT | Weighted FBP of the views rebinned to parallel rays | `FilteredBackProjection(proj, CTGen3SystemMatrix(object_meta, proj_meta), filter="hann", slice_thickness=1.25)()` | [Filtered back projection (CT)](notebooks/t_fbp_ct.ipynb), [DICOM-CT-PD](notebooks/t_CT_GEN3.ipynb) |
| Cone-beam CT | FDK | `FilteredBackProjection(proj, CTConeBeamFlatPanelSystemMatrix(object_meta, proj_meta), filter="ram-lak")()` | [Cone-beam micro-CT](notebooks/t_CT_microct.ipynb) |

</div>
</section>

<section class="pt-arch-panel" id="tab-modality" role="tabpanel" aria-labelledby="t-modality" hidden>
<p class="pt-arch-intro">Each modality has its own files, readers, metadata and system matrix. For statistical iterative reconstruction, all three share the same classes from the likelihood on. Analytic reconstruction skips the likelihood: see the <a href="#overview">Overview</a>.</p>
<div class="pt-arch-bar">
  <div class="pt-arch-seg" id="pt-arch-modality" role="group" aria-label="Show one modality"></div>
  <button type="button" class="pt-arch-btn is-primary" id="pt-arch-walk-start">Walk through one reconstruction</button>
  <p class="pt-arch-hint">Hover a box to follow its path. Click it for details.</p>
</div>
<div id="pt-arch-map" role="group" aria-label="Data, readers, standard objects and system matrix for each modality, then one likelihood, algorithm and reconstruction for all"></div>
<section id="pt-arch-walk" class="pt-arch-walk" aria-live="polite" hidden></section>
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
| A file format | a reader function `read_<format>` | return `(projections, object_meta, proj_meta)` in the <a href="#io-standard">standard form</a> | `io.SPECT.read_simind` |

</div>

<p class="pt-arch-intro">The conventions these parts share (the grid, lengths in mm, the patient frame) are in <a href="#io-standard">Data and conventions</a>. The <a href="contributing/testing.html#contract-tests">contract tests</a> check that a new system matrix's <code>backward</code> is the transpose of its <code>forward</code>.</p>
</section>

<section class="pt-arch-panel pt-io" id="tab-io" role="tabpanel" aria-labelledby="t-io" hidden>
<script>document.documentElement.classList.add("pt-io-js");</script>
<div class="pt-io-views" role="tablist" aria-label="Data and conventions">
  <button type="button" role="tab" id="t-io-map" aria-controls="io-map" aria-selected="true">Map</button>
  <button type="button" role="tab" id="t-io-standard" aria-controls="io-standard" aria-selected="false" tabindex="-1">The standard form</button>
  <button type="button" role="tab" id="t-io-vendors" aria-controls="io-vendors" aria-selected="false" tabindex="-1">Vendor rules</button>
</div>

<div class="pt-io-view" id="io-map" role="tabpanel" aria-labelledby="t-io-map">
<div id="pt-io" class="pt-io-tool" data-src="_static/arch/io-conventions.json"></div>
<div class="pt-io-static">

```{io-map}
```

</div>
</div>

<div class="pt-io-view" id="io-standard" role="tabpanel" aria-labelledby="t-io-standard">

```{io-standard}
```

</div>

<div class="pt-io-view" id="io-vendors" role="tabpanel" aria-labelledby="t-io-vendors">

```{io-vendors}
```

</div>
</section>

<script src="_static/arch/pt-arch-data.js"></script>
<script src="_static/arch/pt-arch-catalog.js"></script>
<script src="_static/arch/pt-arch.js"></script>
<script src="_static/arch/pt-arch-overview.js"></script>
<script src="_static/arch/pt-io.js"></script>
</div>
