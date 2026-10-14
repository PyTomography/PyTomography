"""Reading PET data into PyTomography's standard form: GATE simulations (:func:`read_gate`), list mode in PETSIRD, the
format of the Emission Tomography Standardization Initiative (:func:`read_petsird`; ETSI's ``petsird`` package reads
its header: ``pip install "pytomography[petsird]"``), and GE list mode with GE's corrections (:func:`read_ge_hdf5`,
:func:`read_ge_corrections`).

Every reader returns ``(events, object_meta, proj_meta)``: int32 events (two detector ids, and a TOF bin), the object
grid placed in the patient, and :class:`~pytomography.metadata.PET.PETLMProjMeta` with the scanner's ``info``
(:func:`scanner_info`), whose crystals :func:`scanner_lut` places in the object frame. The sinogram functions work on
any reader's events and ``info``."""
from .scanner import scanner_info, scanner_lut, crystal_pair_index
from .shared import (listmode_to_sinogram, sinogram_to_listmode, smooth_randoms_sinogram, randoms_sinogram_to_sinogramTOF,
                     normalization_sinogram, remove_events_outside, LazySinogram)
from .gate import read_gate, gate_normalization, gate_image
from .petsird import read_petsird
from .ge import read_ge_hdf5, read_ge_corrections
