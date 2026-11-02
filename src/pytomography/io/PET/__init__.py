"""Read PET data into PyTomography's standard form, and bin its list mode events into sinograms.

Every reader returns ``(events, object_meta, proj_meta)``: the events (int32 detector ids, ``[N, 2]``, or ``[N, 3]``
with the TOF bin), the reconstruction grid (:class:`~pytomography.metadata.ObjectMeta`, centred on the scanner's
rings) and a :class:`~pytomography.metadata.PET.PETLMProjMeta` whose ``info`` describes the scanner
(:mod:`pytomography.io.PET.scanner`).

Readers:

* :func:`read_gate`: the coincidences of a GATE simulation (its ROOT output and geometry macro);
* :func:`read_petsird`: list mode in PETSIRD 0.9, the format of the Emission Tomography Standardization Initiative
  (ETSI), whose header ETSI's ``petsird`` package reads (``pip install "pytomography[petsird]"``);
* :func:`read_ge_hdf5` and :func:`read_ge_corrections`: GE list mode (HDF5) and GE's corrections for it.

The scanner:

* :func:`scanner_info`: the ``info`` dictionary of a scanner, built and checked;
* :func:`scanner_lut`: the position of every crystal in the object frame, in mm;
* :func:`crystal_pair_index`: the index of a pair of crystals in ``torch.combinations`` order, the order of every
  reader's sensitivity weights.

Sinograms, from any reader's events and ``info``:

* :func:`listmode_to_sinogram`, :func:`sinogram_to_listmode` and :class:`LazySinogram`;
* :func:`normalization_sinogram`, :func:`smooth_randoms_sinogram` and :func:`randoms_sinogram_to_sinogramTOF`; the
  steps of a sinogram's axes in mm and degrees, :func:`sinogram_spacing`;
* :func:`remove_events_outside`: the events whose LOR crosses the reconstruction grid.

GATE helpers: :func:`gate_normalization` (normalisation weights from a simulated calibration scan) and
:func:`gate_image` (an image in GATE's world, placed in the patient).
"""
from .scanner import scanner_info, scanner_lut, crystal_pair_index
from .shared import (listmode_to_sinogram, sinogram_to_listmode, smooth_randoms_sinogram, randoms_sinogram_to_sinogramTOF,
                     sinogram_spacing,
                     normalization_sinogram, remove_events_outside, LazySinogram)
from .gate import read_gate, gate_normalization, gate_image
from .petsird import read_petsird
from .ge import read_ge_hdf5, read_ge_corrections
