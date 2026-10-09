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

    The results do not depend on the budget. Without one (the default), sinograms are built whole and blocks have fixed sizes.

    Args:
        gb (float | None): Memory budget in GB, or None for no budget.
    """
    global memory_budget
    if gb is not None and gb <= 0:
        raise ValueError(f"the memory budget must be positive, got {gb} GB")
    memory_budget = None if gb is None else float(gb) * 1e9
    
    