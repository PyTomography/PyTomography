from __future__ import annotations
import torch
from torch.nn.functional import pad
from kornia.geometry.transform import rotate
from scipy.ndimage import affine_transform
import numpy as np

def rotate_detector_z(
    x: torch.Tensor,
    angles: torch.tensor,
    mode: str = 'bilinear',
    negative: bool = False):
    r"""Returns an object tensor in a rotated reference frame such that the scanner is located at the +x axis. Note that the scanner angle :math:`\beta` is related to :math:`\phi` (azimuthal angle) by :math:`\phi = 3\pi/2 - \beta`. 

    Args:
        x (torch.tensor[batch_size, Lx, Ly, Lz]): Tensor aligned with cartesian coordinate system specified
        by the manual. 
        angles (torch.Tensor): The angles :math:`\beta` where the scanner is located for each element in the batch x.
        mode (str, optional): Method of interpolation used to get rotated object. Defaults to bilinear.
        negative (bool, optional): If True, applies an inverse rotation. In this case, the tensor
        x is an object in a coordinate system aligned with :math:`\beta`, and the function rotates the
        x back to the original cartesian coordinate system specified by the users manual. In particular, if one
        uses this function on a tensor with negative=False, then applies this function to that returned
        tensor with negative=True, it should return the same tensor. Defaults to False.

    Returns:
        torch.tensor[batch_size, Lx, Ly, Lz]: Rotated tensor.
    """
    phi = 270 - angles
    if not negative:
        x = rotate(x.permute(2,0,1).unsqueeze(0), -phi, mode=mode).squeeze().permute(1,2,0)
    else:
        x = rotate(x.permute(2,0,1).unsqueeze(0), phi, mode=mode).squeeze().permute(1,2,0)
    return x

def compute_pad_size(width: int):
    """Computes the pad width required such that subsequent rotation retains the entire object

    Args:
        width (int): width of the corresponding axis (i.e. number of elements in the dimension)

    Returns:
        int: the number of pixels by which the axis needs to be padded on each side
    """
    return int(np.ceil((np.sqrt(2)*width - width)/2)) 

def compute_pad_size_padded(width: int):
    """Computes the width by which an object was padded, given its padded size.

    Args:
        width (int): width of the corresponding axis (i.e. number of elements in the dimension)

    Returns:
        int: the number of pixels by which the object was padded to get to this width
    """
    # Note: This seemed to empirically work for all integers between 1 and 10million, so I'll use it
    a = (np.sqrt(2) - 1)/2
    if width%2==0:
        width_old = int(2*np.floor((width/2)/(1+2*a)))
    else:
        width_old = int(2*np.floor(((width-1)/2)/(1+2*a)))
    return int((width-width_old)/2)

def pad_object(object: torch.Tensor, mode='constant'):
    """Pads object tensors by enough pixels in the xy plane so that subsequent rotations don't crop out any of the object

    Args:
        object (torch.Tensor[batch_size, Lx, Ly, Lz]): object tensor to be padded
        mode (str, optional): _description_. Defaults to 'constant'.

    Returns:
        _type_: _description_
    """
    pad_size = compute_pad_size(object.shape[-2]) 
    if mode=='back_project':
        # replicate along back projected dimension (x)
        object = pad(object.unsqueeze(0), [0,0,0,0,pad_size,pad_size], mode='replicate').squeeze()
        object = pad(object, [0,0,pad_size,pad_size], mode='constant')
        return object
    else:
        return pad(object, [0,0,pad_size,pad_size,pad_size,pad_size], mode=mode)

def unpad_object(object: torch.Tensor):
    """Unpads a padded object tensor in the xy plane back to its original dimensions

    Args:
        object (torch.Tensor[batch_size, Lx', Ly', Lz]): padded object tensor

    Returns:
        torch.Tensor[batch_size, Lx, Ly, Lz]: Object tensor back to it's original dimensions.
    """
    pad_size = compute_pad_size_padded(object.shape[-2])
    return object[pad_size:-pad_size,pad_size:-pad_size,:]

def pad_proj(proj: torch.Tensor, mode: str = 'constant', value: float = 0):
    """Pads projections along the Lr axis

    Args:
        proj (torch.Tensor[batch_size, Ltheta, Lr, Lz]): Projections tensor.
        mode (str, optional): Padding mode to use. Defaults to 'constant'.
        value (float, optional): If padding mode is constant, fill with this value. Defaults to 0.

    Returns:
        torch.Tensor[batch_size, Ltheta, Lr', Lz]: Padded projections tensor.
    """
    pad_size = compute_pad_size(proj.shape[-2])  
    return pad(proj, [0,0,pad_size,pad_size], mode=mode, value=value)

def unpad_proj(proj: torch.Tensor):
    """Unpads the projections back to original Lr dimensions

    Args:
        proj (torch.Tensor[batch_size, Ltheta, Lr', Lz]): Padded projections tensor

    Returns:
        torch.Tensor[batch_size, Ltheta, Lr, Lz]: Unpadded projections tensor
    """
    pad_size = compute_pad_size_padded(proj.shape[-2])
    return proj[:,pad_size:-pad_size,:]

def pad_object_z(object: torch.Tensor, pad_size: int, mode='constant'):
    """Pads an object tensor along z. Useful for PSF modeling 

    Args:
        object (torch.Tensor[batch_size, Lx, Ly, Lz]): Object tensor
        pad_size (int): Amount by which to pad in -z and +z
        mode (str, optional): Padding mode. Defaults to 'constant'.

    Returns:
        torch.Tensor[torch.Tensor[batch_size, Lx, Ly, Lz']]: Padded object tensor along z.
    """
    return pad(object, [pad_size,pad_size,0,0,0,0], mode=mode)

def unpad_object_z(object: torch.Tensor, pad_size: int):
    """Unpads an object along the z dimension

    Args:
        object (torch.Tensor[batch_size, Lx, Ly, Lz']): Padded object tensor along z.
        pad_size (int): Amount by which the padded tensor was padded in the z direcion

    Returns:
        torch.Tensor[batch_size, Lx, Ly, Lz]:Unpadded object tensor.
    """
    
    return object[:,:,pad_size:-pad_size]
