---
html_theme.sidebar_secondary.remove: true
---

# PyTomography

<div class="pt-landing">

<section class="pt-hero">
  <div class="pt-hero-text">
    <p class="pt-eyebrow">Open source · GPU accelerated · 4.0</p>
    <p class="pt-h1">Quantitative SPECT, PET and CT reconstruction</p>
    <div class="pt-install"><span class="pt-prompt">$</span><code>pip install pytomography</code><button type="button" class="pt-copy" data-copy="pip install pytomography">Copy</button></div>
  </div>
  <figure class="pt-hero-fig">
    <video src="_static/landing/mip.mp4" poster="_static/landing/mip_poster.webp" autoplay muted loop playsinline preload="auto" width="600" height="900" role="img" aria-label="A Lu-177-PSMA patient, upright, turning about the body's long axis: a maximum intensity projection of the SPECT, reconstructed with PyTomography from two bed positions on a GE Discovery 670 Pro (OSEM, 4 iterations of 8 subsets, with CT attenuation, collimator PSF and triple-energy-window scatter correction) and stitched, in PyTomography's PET colour scale over a maximum intensity projection of the patient's CT"></video>
    <figcaption><span>Lu-177-PSMA patient · GE Discovery 670 Pro, two beds</span><span>OSEM 4×8, attenuation, PSF, TEW scatter</span></figcaption>
  </figure>
</section>

<section class="pt-modalities" aria-label="Supported modalities">
  <div><b>SPECT</b><span>DICOM from GE, Siemens and StarGuide, and SIMIND. Multiple beds and photopeaks, cardiac reorientation, hybrid Monte Carlo scatter.</span><button type="button" class="pt-sample" data-ptv-sample="t_dicommultibed">See sample</button></div>
  <div><b>PET</b><span>GATE sinograms and list mode, time of flight, GE Discovery MI and PETSIRD. Randoms and single scatter simulation.</span><button type="button" class="pt-sample" data-ptv-sample="t_PETGATE_DIP">See sample</button></div>
  <div><b>CT</b><span>Raw DICOM-CT-PD projections from 3rd-generation clinical scanners, and cone-beam micro-CT. Helical filtered back projection, FDK and OS-SART.</span><button type="button" class="pt-sample" data-ptv-sample="t_CT_GEN3">See sample</button></div>
  <div><b>Your own system</b><span>Every system matrix is a forward and back projector you can replace, so new geometries, priors and networks reuse every algorithm.</span></div>
</section>

<section class="pt-split">
<div class="pt-split-text">
  <p class="pt-eyebrow">How it works</p>
  <h2 class="pt-h2">Every reconstruction is three steps</h2>
  <p>Read the data, describe the physics, then choose a likelihood and an algorithm. Each step is its own object, so you can swap one without touching the others.</p>
  <dl class="pt-steps">
    <dt>1 · Data</dt><dd>Projections, energy windows and geometry, read from the file the scanner wrote.</dd>
    <dt>2 · Physics</dt><dd>Attenuation from the CT and the collimator-detector response, applied inside the projector.</dd>
    <dt>3 · Statistics</dt><dd>A Poisson likelihood with scatter as an additive term, solved by the iterative algorithm of your choice. Filtered back projection needs no likelihood.</dd>
  </dl>
</div>
<div class="pt-split-code">

```python
from pytomography import io
from pytomography.transforms.SPECT import (
    SPECTAttenuationTransform, SPECTPSFTransform)
from pytomography.projectors.SPECT import (
    SPECTSystemMatrix)
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM

# 1. Data: Lu-177 SPECT/CT, every energy window
data = io.SPECT.read_dicom(file_NM)
projections, object_meta, proj_meta = data
photopeak = projections[0]
scatter = io.SPECT.energy_window_scatter(
    projections, proj_meta, peak=0, lower=1, upper=2)

# 2. Physics, in mm and keV (208 keV here)
photon_energy = proj_meta.energy_windows[0].photon_energy
mu = io.attenuation_map(files_CT, object_meta,
                        energy=photon_energy)  # 1/mm
attenuation = SPECTAttenuationTransform(mu)
psf_meta = io.SPECT.collimator_psf(
    "SY-ME", photon_energy, intrinsic_fwhm=3.8)
psf = SPECTPSFTransform(psf_meta)
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[attenuation, psf],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

# 3. Statistics and algorithm
likelihood = PoissonLogLikelihood(
    system_matrix, photopeak, scatter)
recon = OSEM(likelihood)(n_iters=4, n_subsets=8)
```

</div>
</section>

<section class="pt-join">
  <div>
    <p class="pt-eyebrow">Built by its users</p>
    <h2 class="pt-h2">Add the method your research needs</h2>
    <p>PyTomography grows through contributions from the people who use it. The feature board lists work we want done, each item with data to test on, a place in the code to start, and a clear definition of done.</p>
  </div>
  <div class="pt-join-links">
    <a href="contributing/feature-board.html"><b>Feature board</b><span>Pick something to build</span></a>
    <a href="contributing/index.html"><b>Contributing guide</b><span>Set up, test, open a pull request</span></a>
    <a href="ai.html"><b>For AI assistants</b><span>llms.txt and Markdown pages</span></a>
  </div>
</section>

<section class="pt-cite">
<div>
  <p class="pt-eyebrow">Citing PyTomography</p>
  <p>If PyTomography helps your research, please cite the paper in <a href="https://doi.org/10.1016/j.softx.2024.102020">SoftwareX</a>.</p>
</div>
<div class="pt-cite-code">

```bibtex
@article{polson2025pytomography,
  title   = {PyTomography: A python library for medical image reconstruction},
  author  = {Polson, Lucas A. and Fedrigo, Roberto and Li, Chenguang and Sabouri, Maziar and
             Dzikunu, Obed and Ahamed, Shadab and Karakatsanis, Nicolas and Kurkowska, Sara and
             Sheikhzadeh, Peyman and Esquinas, Pedro and Rahmim, Arman and Uribe, Carlos},
  journal = {SoftwareX},
  volume  = {29},
  pages   = {102020},
  year    = {2025},
  doi     = {10.1016/j.softx.2024.102020}
}
```

</div>
</section>

</div>

```{toctree}
:hidden:

install
tutorials/index
architecture
API <api/pytomography/index>
Contribute <contributing/index>
SPECT parameters <spect_parameters>
```
