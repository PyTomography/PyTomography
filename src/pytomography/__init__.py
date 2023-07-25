import torch
device = torch.device("cpu")
dtype = torch.float32
delta = 1e-11

def set_dtype(dt):
    global dtype
    global delta
    dtype = dt
    torch.set_default_dtype(dt)
    if dt==torch.float16:
        delta = 1e-6
    elif dt==torch.float32:
        delta = 1e-11
    
def set_device(d):
    global device
    device = d
    
    