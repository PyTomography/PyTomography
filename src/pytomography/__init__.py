import torch
import os
import shutil
import sys
from importlib.metadata import version

__version__: str = version('pytomography')

# Silence parallelproj import
os.environ['PARALLELPROJ_SILENT_IMPORT'] = '1'

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

GPU_CHECK_URL = "https://pytomography.readthedocs.io/en/latest/ai.html#check-that-pytomography-uses-the-gpu"


def _device_message(device: torch.device, nvidia_smi: str | None, cuda_visible_devices: str | None) -> str | None:
    """What PyTomography says at import about the device it found.

    A machine with an NVIDIA GPU (``nvidia-smi`` installed) where PyTorch cannot use it gets a warning with the likely
    cause: a broken install runs on the CPU without failing, many times slower, and is easily taken for PyTomography's
    normal speed.

    Args:
        device (torch.device): The device PyTomography found.
        nvidia_smi (str | None): Path of ``nvidia-smi``, or None if it is not installed.
        cuda_visible_devices (str | None): The ``CUDA_VISIBLE_DEVICES`` environment variable, or None if unset.

    Returns:
        str | None: The message, or None on a CUDA GPU.
    """
    if device.type == "cuda":
        return None
    if device.type == "mps":
        return "PyTomography found an Apple Silicon GPU (mps); support for it is experimental."
    if nvidia_smi is None:
        return "PyTomography did not find a GPU on this machine, so it runs on the CPU."
    if cuda_visible_devices is not None and cuda_visible_devices.strip() in ("", "-1"):
        cause = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} hides the GPU; unset it to use the GPU."
    elif torch.version.cuda is None:
        cause = (f"PyTorch {torch.__version__} is a CPU-only build: reinstall PyTorch with the command for your CUDA "
                 "version from https://pytorch.org/get-started/locally/.")
    else:
        cause = (f"PyTorch {torch.__version__} is built for CUDA {torch.version.cuda} but cannot use the GPU: the NVIDIA "
                 "driver may be too old for it (nvidia-smi shows the newest CUDA version the driver supports).")
    return ("WARNING: this machine has an NVIDIA GPU, but PyTomography is running on the CPU, which is many times "
            f"slower. {cause} See {GPU_CHECK_URL}")


def _parallelproj_message(device: torch.device | str, cuda_enabled: bool) -> str | None:
    """What PyTomography says when PET or CT is imported with parallelproj's CPU build although PyTorch uses a CUDA
    GPU: the PET and CT projectors then run on the CPU, many times slower.

    Args:
        device (torch.device | str): PyTomography's device.
        cuda_enabled (bool): ``parallelproj_core.cuda_enabled``.

    Returns:
        str | None: The message, or None when there is nothing to say.
    """
    if torch.device(device).type != "cuda" or cuda_enabled:
        return None
    return ("WARNING: PyTomography uses the GPU, but parallelproj is its CPU build, so PET and CT projections run on the "
            "CPU, many times slower. Install the libparallelproj build that matches your CUDA version, e.g. "
            '`conda install -c conda-forge "libparallelproj=*=cuda130*" parallelproj`. See ' + GPU_CHECK_URL)


def _say(message: str | None) -> None:
    if message:
        print(message, file=sys.stderr if message.startswith("WARNING") else sys.stdout)


_parallelproj_checked = False


def _check_parallelproj(parallelproj_core) -> None:
    """Warns once, when PET or CT is first imported, if parallelproj is its CPU build on a GPU machine."""
    global _parallelproj_checked
    if not _parallelproj_checked:
        _parallelproj_checked = True
        _say(_parallelproj_message(device, bool(getattr(parallelproj_core, "cuda_enabled", False))))


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
_say(_device_message(device, shutil.which("nvidia-smi"), os.environ.get("CUDA_VISIBLE_DEVICES")))
dtype = torch.float32
delta = 1e-11
verbose = False

def set_dtype(dt: float):
    global dtype
    global delta
    dtype = dt
    torch.set_default_dtype(dt)
    if dt==torch.float16:
        delta = 1e-5
    elif dt==torch.float32:
        delta = 1e-11
    
def set_device(d: str):
    global device
    device = d
    
def set_verbose(b: bool):
    global verbose
    verbose = b

#: Host memory (RAM), in bytes, that PyTomography's memory-heavy steps plan for, or None for no budget. Set it with
#: :func:`set_memory_budget`.
memory_budget = None

def set_memory_budget(gb: float | None):
    """Sets how much host memory (RAM) PyTomography's memory-heavy steps may use, in GB. They split their work to fit:

    * steps that go through every pair of crystals of a PET scanner (normalization weights and sinograms, the list mode
      sensitivity image; 411 million pairs for the Siemens Biograph mMR) do so in blocks sized from the budget;
    * a sinogram that would take more than a quarter of the budget is built one subset of angles at a time instead of
      whole (a :class:`~pytomography.io.PET.shared.LazySinogram`), e.g. a time of flight sinogram;
    * a reconstruction whose subsets would not fit stops before it starts, with the number of subsets that would.

    The results do not depend on the budget. Without one (the default), blocks have fixed sizes, and sinograms are built whole unless one would take more than 8 GB (a TOF sinogram of a clinical scanner).

    To see what a reconstruction will need before running it, ask its system matrix or algorithm: ``estimate_memory``
    gives the peak RAM and GPU memory for given settings (e.g. ``n_subsets``), and ``plan_memory`` the fastest settings
    that fit a RAM and GPU budget, by default what is free on this machine (see :mod:`pytomography.utils.memory`).

    Args:
        gb (float | None): Memory budget in GB, or None for no budget.
    """
    global memory_budget
    if gb is not None and gb <= 0:
        raise ValueError(f"the memory budget must be positive, got {gb} GB")
    memory_budget = None if gb is None else float(gb) * 1e9
    
    