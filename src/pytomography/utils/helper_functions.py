from __future__ import annotations
import torch
from torch.nn.functional import pad
from kornia.geometry.transform import rotate
import numpy as np
import pydicom

def rev_cumsum(x: torch.Tensor):
    """Reverse cumulative sum along the first axis of a tensor of shape [batch_size, Lx, Ly, Lz].
    since this is used with CT correction, the initial voxel only contributes 1/2.

    Args:
        x (torch.tensor[batch_size,Lx,Ly,Lz]): Tensor to be summed

    Returns:
        torch.tensor[batch_size, Lx, Ly, Lz]: The cumulatively summed tensor.
    """
    return torch.cumsum(x.flip(dims=(1,)), dim=1).flip(dims=(1,)) - x/2

def rotate_detector_z(
    x: torch.Tensor,
    angles: torch.tensor,
    mode: str = 'bilinear',
    negative: bool = False):
    """Returns an object tensor in a rotated reference frame such that the scanner is located at the +x axis. Note that the scanner angle $\beta$ is related to $\phi$ (azimuthal angle) by $\phi = 3\pi/2 - \beta$. 

    Args:
        x (torch.tensor[batch_size, Lx, Ly, Lz]): Tensor aligned with cartesian coordinate system specified
        by the manual. 
        angles (torch.Tensor): The angles $\beta$ where the scanner is located for each element in the batch x.
        mode (str, optional): Method of interpolation used to get rotated image. Defaults to bilinear.
        negative (bool, optional): If True, applies an inverse rotation. In this case, the tensor
        x is an object in a coordinate system aligned with $\beta$, and the function rotates the
        x back to the original cartesian coordinate system specified by the users manual. In particular, if one
        uses this function on a tensor with negative=False, then applies this function to that returned
        tensor with negative=True, it should return the same tensor. Defaults to False.

    Returns:
        torch.tensor[batch_size, Lx, Ly, Lz]: Rotated tensor.
    """
    phi = 270 - angles
    if not negative:
        x = rotate(x.permute(0,3,1,2), -phi, mode=mode).permute(0,2,3,1)
    else:
        x = rotate(x.permute(0,3,1,2), phi, mode=mode).permute(0,2,3,1)
    return x


def get_distance(Lx: int, r: float, dx: float):
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

def compute_pad_size(width: int):
    """Computes the pad width required such that subsequent rotation retains the entire image

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
        object = pad(object, [0,0,0,0,pad_size,pad_size], mode='replicate')
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
    return object[:,pad_size:-pad_size,pad_size:-pad_size,:]

def pad_image(image: torch.Tensor, mode: str = 'constant', value: float = 0):
    """Pads an image along the Lr axis

    Args:
        image (torch.Tensor[batch_size, Ltheta, Lr, Lz]): Image tensor.
        mode (str, optional): Padding mode to use. Defaults to 'constant'.
        value (float, optional): If padding mode is constant, fill with this value. Defaults to 0.

    Returns:
        torch.Tensor[batch_size, Ltheta, Lr', Lz]: Padded image tensor.
    """
    pad_size = compute_pad_size(image.shape[-2])  
    return pad(image, [0,0,pad_size,pad_size], mode=mode, value=value)

def unpad_image(image: torch.Tensor):
    """Unpads the image back to original Lr dimensions

    Args:
        image (torch.Tensor[batch_size, Ltheta, Lr', Lz]): Padded image tensor

    Returns:
        torch.Tensor[batch_size, Ltheta, Lr, Lz]: Unpadded image tensor
    """
    pad_size = compute_pad_size_padded(image.shape[-2])
    return image[:,:,pad_size:-pad_size,:]

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
    
    return object[:,:,:,pad_size:-pad_size]

def get_object_nearest_neighbour(object: torch.Tensor, shifts: list[int]):
    neighbour = pad(object, [1,1,1,1,1,1])
    neighbour = torch.roll(neighbour, shifts=shifts, dims=(1,2,3))
    return neighbour[:,1:-1,1:-1,1:-1]

def get_blank_below_above(image: torch.tensor):
    """Obtains the number of blank z-slices at the sup (``blank_above``) and inf (``blank_below``) of the projection data. This method is entirely empircal, and looks for z slices where there are zero detected counts.

    Args:
        image (torch.tensor): Image data from a scanner

    Returns:
        Sequence[int]: A tuple of two elements corresponding to the number of blank slices at the inf, and the number of blank slices at the sup.
    """
    greater_than_zero = image[0].cpu().numpy().sum(axis=(0,1)) > 0
    blank_below = np.argmax(greater_than_zero)
    blank_above = image[0].cpu().numpy().shape[-1] - np.argmax(greater_than_zero[::-1])
    return blank_below, blank_above

def bilinear_transform(
    arr: np.array,
    a1: float,
    b1: float,
    a2:float ,
    b2:float
    ) -> np.array:
    """Converts an array of Hounsfield Units into linear attenuation coefficient using the bilinear transformation :math:`f(x)=a_1x+b_1` for positive :math:`x` and :math:`f(x)=a_2x+b_2` for negative :math:`x`.

    Args:
        arr (np.array): Array to be transformed using bilinear transformation
        a1 (float): Bilinear slope for negative input values
        b1 (float): Bilinear intercept for negative input values
        a2 (float): Bilinear slope for positive input values
        b2 (float): Bilinear intercept for positive input values

    Returns:
        np.array: Transformed array.
    """
    arr_transform = np.piecewise(
        arr,
        [arr <= 0, arr > 0],
        [lambda x: a1*x + b1,
        lambda x: a2*x + b2]
    )
    arr_transform[arr_transform<0] = 0
    return arr_transform


    
