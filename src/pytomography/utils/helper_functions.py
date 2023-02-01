import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np

# cumulative sum, but initial voxel only contriubtes 1/2 mu dx
# x: [batch_size, Lx, Ly, Lz]
def rev_cumsum(x):
    return torch.cumsum(x.flip(dims=(1,)), dim=1).flip(dims=(1,)) - x/2
    #return torch.cumsum(x, dim=1) - x/2

# Rotates the scanner scanner around an object of
# [batch_size, Lx, Ly, Lz] by angle theta in object space
# about the z axis. This is a bit tricky to understand.
# angle = beta. Rotating detector beta corresponds to rotating
# patient by -phi where phi = 3pi/2 - beta. Inverse rotatation 
# is rotating by phi (needed for back proijection)
def rotate_detector_z(x, angle, interpolation = InterpolationMode.BILINEAR, negative=False):
    phi = 270 - angle
    if not negative:
        return rotate(x.permute(0,3,1,2), -phi, interpolation).permute(0,2,3,1)
    else:
        return rotate(x.permute(0,3,1,2), phi, interpolation).permute(0,2,3,1)



def get_distance(N, r, dx):
    if N%2==0:
        d = r + (N//2 - np.arange(N)) * dx
    else:
        d = r + (N//2 - np.arange(N) - 1/2) * dx
    # Correction for if radius of scanner is inside the the bounds
    d[d<0] = 0
    return d