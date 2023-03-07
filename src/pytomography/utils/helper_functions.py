import torch
from torch.nn.functional import pad
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np

def rev_cumsum(x):
    """Reverse cumulative sum along the first axis of a tensor of shape [batch_size, Lx, Ly, Lz].
    since this is used with CT correction, the initial voxel only contributes 1/2.

    Args:
        x (torch.tensor[batch_size,Lx,Ly,Lz]): Tensor to be summed

    Returns:
        torch.tensor[batch_size, Lx, Ly, Lz]: The cumulatively summed tensor.
    """
    return torch.cumsum(x.flip(dims=(1,)), dim=1).flip(dims=(1,)) - x/2

# Rotates the scanner scanner around an object of
# [batch_size, Lx, Ly, Lz] by angle theta in object space
# about the z axis. This is a bit tricky to understand.
# angle = beta. Rotating detector beta corresponds to rotating
# patient by -phi where phi = 3pi/2 - beta. Inverse rotatation 
# is rotating by phi (needed for back proijection)
def rotate_detector_z(x, angle, interpolation = InterpolationMode.BILINEAR, negative=False):
    """Returns an object tensor in a rotated reference frame such that the scanner is located at the +x axis. Note that the scanner angle $\beta$ is related to $\phi$ (azimuthal angle) by $\phi = 3\pi/2 - \beta$. 

    Args:
        x (torch.tensor[batch_size, Lx, Ly, Lz]): Tensor aligned with cartesian coordinate system specified
        by the manual. 
        angle (float): The angle $\beta$ where the scanner is located.
        interpolation (InterpolationMode, optional): Method of interpolation used to get rotated image.
        Defaults to InterpolationMode.BILINEAR.
        negative (bool, optional): If True, applies an inverse rotation. In this case, the tensor
        x is an object in a coordinate system aligned with $\beta$, and the function rotates the
        x back to the original cartesian coordinate system specified by the users manual. In particular, if one
        uses this function on a tensor with negative=False, then applies this function to that returned
        tensor with negative=True, it should return the same tensor. Defaults to False.

    Returns:
        torch.tensor[batch_size, Lx, Ly, Lz]: Rotated tensor.
    """
    phi = 270 - angle
    if not negative:
        x = rotate(x.permute(0,3,1,2), -phi, interpolation).permute(0,2,3,1)
    else:
        x = rotate(x.permute(0,3,1,2), phi, interpolation).permute(0,2,3,1)
    return x


def get_distance(Lx, r, dx):
    """Given the radial distance to center of object space from the scanner, computes the distance 
      between each parallel plane (i.e. (y-z plane)) and a detector located at +x. This function is used for point spread function (PSF) blurring where the amount of blurring depends on thedistance from the detector.

    Args:
        Lx (int): The number of y-z planes to compute the distance of
        r (float): The radial distance between the central y-z plane and the detector at +x.
        dx (float): The spacing between y-z planes in Euclidean distance.

    Returns:
        np.array[Lx]: An array of distances for each y-z plane to the detector.
    """
    if Lx%2==0:
        d = r + (Lx/2 - np.arange(Lx) - 1/2) * dx
    else:
        d = r + ((Lx-1)/2 - np.arange(Lx) ) * dx
    # Correction for if radius of scanner is inside the the bounds
    d[d<0] = 0
    return d

def compute_pad_size(width):
    return int(np.ceil((np.sqrt(2)*width - width)/2)) 

def pad_object(object, mode='constant'):
    pad_size = compute_pad_size(object.shape[-2]) 
    if mode=='back_project':
        # replicate along back projected dimension (x)
        object = pad(object, [0,0,0,0,pad_size,pad_size], mode='replicate')
        object = pad(object, [0,0,pad_size,pad_size], mode='constant')
        return object
    else:
        return pad(object, [0,0,pad_size,pad_size,pad_size,pad_size], mode=mode)

def unpad_object(object, original_shape):
    pad_size = (object.shape[-2] - original_shape[-2])//2 
    return object[:,pad_size:-pad_size,pad_size:-pad_size,:]

def pad_image(image, mode='constant', value=0):
    pad_size = compute_pad_size(image.shape[-2])  
    return pad(image, [0,0,pad_size,pad_size], mode=mode, value=value)

def unpad_image(image, original_shape):
    pad_size = (image.shape[-2] - original_shape[-2])//2 
    return image[:,:,pad_size:-pad_size,:]

def paded_PSF_adjustment(object):
    pad_size = compute_pad_size(object.shape[-2])
    return pad(object, [0,0,pad_size,pad_size,pad_size,pad_size])

def pad_object_z(object, pad_size, mode='constant'):
    return pad(object, [pad_size,pad_size,0,0,0,0])

def unpad_object_z(object, pad_size):
    return object[:,:,:,pad_size:-pad_size]