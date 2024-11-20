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
    
    