import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("PyTomography did not find a GPU available on this machine. If this is not expected, please check your CUDA installation.")
dtype = torch.float32
delta = 1e-11

from importlib.metadata import version
__version__: str = version('pytomography')

def set_dtype(dt):
    global dtype
    global delta
    dtype = dt
    torch.set_default_dtype(dt)
    if dt==torch.float16:
        delta = 1e-5
    elif dt==torch.float32:
        delta = 1e-11
    
def set_device(d):
    global device
    device = d
    
    