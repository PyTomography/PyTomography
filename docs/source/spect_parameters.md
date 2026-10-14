---
html_theme.sidebar_secondary.remove: true
---

# SPECT parameters

<p class="pt-par-lede">The collimators, detectors and cameras PyTomography knows by name. Each call returns plain numbers, which spectarf takes to make the angular response functions of a PSF (<a href="notebooks/t_arf_psf.html">ARF tutorial</a>).</p>

```python
from pytomography.io import SPECT
collimator = SPECT.collimator("SY-HE")         # hole diameter, septa, length, shape: the Collimators tab
detector = SPECT.detector("NaI-9.5mm")         # crystal, cover, backscatter: the Detectors tab
camera = SPECT.camera("Siemens Symbia T")      # energy and intrinsic resolution: the Cameras tab
```

Any value can be given explicitly instead, for a camera that is not listed.

```{spect-parameters}
```

<script>
document.querySelectorAll('.pt-par-filter').forEach(function (input) {
  var table = input.parentElement.querySelector('table') || input.closest('.sd-tab-content').querySelector('table');
  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase();
    table.querySelectorAll('tbody tr').forEach(function (row) {
      row.style.display = !q || row.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
    });
  });
});
</script>
