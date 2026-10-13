"""Quantitative SPECT, PET and CT reconstruction.

PyTomography reconstructs images from projection data. Data goes through four kinds of objects:

* :mod:`pytomography.io` reads raw data (DICOM, SIMIND, GATE, PETSIRD, vendor formats) into PyTomography's standard
  form: the measured data, an ``object_meta`` (the image grid, in mm, placed in the patient) and a ``proj_meta``
  (the acquisition), and saves images;
* :mod:`pytomography.projectors` holds the system matrices, which map an object to projections (``forward``) and
  projections back to the object (``backward``, the back projection), built from the
  :mod:`pytomography.transforms` that model attenuation, resolution and so on;
* :mod:`pytomography.likelihoods` compares predicted and measured projections;
* :mod:`pytomography.algorithms` reconstructs, optionally with :mod:`pytomography.priors` and
  :mod:`pytomography.callbacks`.

The settings below apply to the whole library: ``device``, the device the computations run on (CUDA when available,
else Apple's MPS, else the CPU; change it with :func:`set_device`), the floating point type and a host memory budget
for the memory-heavy steps.
"""
import torch
import os
import sys
from importlib.metadata import version

__version__: str = version('pytomography')

# Silence parallelproj import
os.environ['PARALLELPROJ_SILENT_IMPORT'] = '1'

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device == "cpu":
    print("PyTomography did not find a GPU available on this machine. If this is not expected, please check your CUDA installation.")
elif str(device).strip() == "mps":
    print("PyTomography found Apple Silicon GPUs, this is experimental")
dtype = torch.float32
"""The floating point type of PyTomography's tensors. Set it with :func:`set_dtype`."""
delta = 1e-11
"""A small number added to denominators to avoid dividing by zero (set by :func:`set_dtype`)."""
verbose = False
"""A flag set by :func:`set_verbose`. No part of the library reads it at present."""

def set_dtype(dt: float):
    """Sets the floating point type of PyTomography's tensors, and torch's default type.

    Also sets :data:`delta` to suit the type: 1e-5 for ``torch.float16``, 1e-11 for ``torch.float32`` (other types
    leave it unchanged).

    Args:
        dt (torch.dtype): The type, e.g. ``torch.float32``.
    """
    global dtype
    global delta
    dtype = dt
    torch.set_default_dtype(dt)
    if dt==torch.float16:
        delta = 1e-5
    elif dt==torch.float32:
        delta = 1e-11

def set_device(d: str):
    """Sets the device PyTomography computes on.

    Args:
        d (str | torch.device): The device, e.g. ``'cuda'``, ``'cuda:1'`` or ``'cpu'``.
    """
    global device
    device = d

def set_verbose(b: bool):
    """Sets the :data:`verbose` flag.

    Args:
        b (bool): The new value.
    """
    global verbose
    verbose = b

memory_budget = None
"""Host memory (RAM), in bytes, that PyTomography's memory-heavy steps plan for, or None for no budget. Set it with
:func:`set_memory_budget`."""

def set_memory_budget(gb: float | None):
    """Sets how much host memory (RAM), in GB, PyTomography's memory-heavy steps may use.

    They split their work to fit:

    * steps that go through every pair of crystals of a PET scanner (normalization weights and sinograms, the list mode
      sensitivity image; 411 million pairs for the Siemens Biograph mMR) do so in blocks sized from the budget;
    * a sinogram that would take more than a quarter of the budget is built one subset of angles at a time instead of
      whole (a :class:`~pytomography.io.PET.LazySinogram`), e.g. a time of flight sinogram;
    * helical CT filtered back projection rebins a chunk of views at a time, at most an eighth of the budget;
    * a reconstruction whose subsets would not fit stops before it starts, with the number of subsets that would.

    The results do not depend on the budget. Without one (the default), blocks have fixed sizes, and sinograms are
    built whole unless one would take more than 8 GB (a TOF sinogram of a clinical scanner).

    To see what a reconstruction will need before running it, ask its system matrix or algorithm: ``estimate_memory``
    gives the peak RAM and GPU memory for given settings (e.g. ``n_subsets``), and ``plan_memory`` the fastest settings
    that fit a RAM and GPU budget, by default what is free on this machine (see :mod:`pytomography.utils.memory`).

    Args:
        gb (float | None): Memory budget in GB, or None for no budget.

    Raises:
        ValueError: If ``gb`` is not positive.
    """
    global memory_budget
    if gb is not None and gb <= 0:
        raise ValueError(f"the memory budget must be positive, got {gb} GB")
    memory_budget = None if gb is None else float(gb) * 1e9
    
    