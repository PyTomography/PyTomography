"""Utilities shared by PyTomography's modalities: memory planning, FBP filters, scatter estimates and smaller helpers.

The main public objects:

* :class:`~pytomography.utils.MemoryEstimate` and :class:`~pytomography.utils.MemoryPlan`: what the ``estimate_memory``
  and ``plan_memory`` methods return; :func:`~pytomography.utils.available_memory` and
  :func:`~pytomography.utils.gpu_budget`: the RAM and GPU memory a computation may use;
  :class:`~pytomography.utils.PeakMemory`: the peak GPU memory of a block of code (module
  :mod:`pytomography.utils.memory`).
* :func:`~pytomography.utils.get_fbp_filter`, :func:`~pytomography.utils.ramp_filter` and the windows of filtered back
  projection, such as :class:`~pytomography.utils.HannFilter` (module :mod:`pytomography.utils.fourier_filters`).
* :func:`~pytomography.utils.compute_EW_scatter`: the dual and triple energy window scatter estimates of SPECT, which
  :func:`pytomography.io.SPECT.energy_window_scatter` computes from the readers' energy windows.
* :mod:`pytomography.utils.sss`: the single scatter simulation of PET
  (:func:`~pytomography.utils.sss.get_sss_scatter_estimate`).
* :mod:`pytomography.utils.simind_mc`: runs the SIMIND Monte Carlo program for
  :class:`~pytomography.projectors.SPECT.MonteCarloHybridSPECTSystemMatrix`.
* :mod:`pytomography.utils.plot_utils`: the PET colour map ``'pet'`` (registered with matplotlib when this package is
  imported) and :func:`~pytomography.utils.plot_utils.dual_imshow`, which overlays two images placed by their affines.
* :func:`~pytomography.utils.rotate_detector_z`, :func:`~pytomography.utils.pad_object` and the other rotation and
  padding functions of :mod:`pytomography.utils.spatial`.
* Smaller helpers: :func:`~pytomography.utils.rev_cumsum`, :func:`~pytomography.utils.get_distance`,
  :func:`~pytomography.utils.get_object_nearest_neighbour`, :func:`~pytomography.utils.get_1d_gaussian_kernel`,
  :func:`~pytomography.utils.print_collimator_parameters` and NIST's attenuation data
  (:func:`~pytomography.utils.get_mu_from_spectrum_interp`).

``import pytomography.utils`` does not import the modules :mod:`~pytomography.utils.sss`,
:mod:`~pytomography.utils.simind_mc` and :mod:`~pytomography.utils.cardiac_spect` (reorienting cardiac SPECT images):
import them by name, e.g. ``from pytomography.utils import sss``.
"""
from .misc import rev_cumsum, get_distance, get_object_nearest_neighbour, print_collimator_parameters, check_if_class_contains_method, get_1d_gaussian_kernel
from .spatial import rotate_detector_z, compute_pad_size, pad_proj, pad_object, unpad_proj, unpad_object, pad_object_z, unpad_object_z
from .nist_data import dual_sqrt_exponential, get_E_mu_data_from_datasheet, get_mu_from_spectrum_interp
from .scatter import compute_EW_scatter
from .fourier_filters import FBPFilter, RamLakFilter, SheppLoganFilter, HannFilter, GeneralizedHammingFilter, CosineFilter, TabulatedFilter, get_fbp_filter, ramp_filter
from .memory import gpu_budget, PeakMemory, MemoryEstimate, MemoryPlan, available_memory
from . import plot_utils  # registers the colormap 'pet' with matplotlib
