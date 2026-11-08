"""Rotation and padding of objects and projections, as the rotate-and-sum SPECT projectors use them, in plain PyTorch.

* :func:`rotate_detector_z` and :func:`rotate_cached`: rotations about z, with sampling grids cached per angle;
* :func:`pad_object`, :func:`pad_proj`, :func:`pad_object_z` and their inverses: the padding that keeps a rotated
  object inside its grid (:func:`compute_pad_size`).

The rotations are bit-identical to those of kornia, which PyTomography used before 4.0.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.nn.functional import pad
from scipy.ndimage import affine_transform
import numpy as np

_ROTATION_GRIDS: dict = {}

def _pixel_to_grid(H: int, W: int, like: torch.Tensor) -> torch.Tensor:
    """[1, 3, 3]: pixel coordinates (x along W, y along H) to the [-1, 1] coordinates of ``grid_sample`` with ``align_corners=True``. Built in float32 and then cast to ``like``'s dtype and device, as kornia (which PyTomography used before 4.0) built it, so that the grids are bit-identical to kornia's."""
    N = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    N[0, 0] = N[0, 0] * 2.0 / (1e-14 if W == 1 else W - 1.0)
    N[1, 1] = N[1, 1] * 2.0 / (1e-14 if H == 1 else H - 1.0)
    return N.unsqueeze(0).to(like)

def affine_grid_pixels(M: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """The sampling grid of affine maps in pixel coordinates, for ``grid_sample(..., align_corners=True)``.

    The image warped by ``M`` holds at output pixel ``p`` the input at ``M^{-1} p``: ``M`` maps input pixel
    coordinates (x, y), x along the last axis (W) and y along H, to output ones. The operations, their order and their
    precision are those of ``kornia.geometry.transform.warp_affine`` (which PyTomography used before 4.0), so the grid,
    and an image sampled with it, are bit-identical to kornia's.

    Args:
        M (torch.Tensor): [B, 2, 3] affine maps.
        H (int): Height of the image (and of the output).
        W (int): Width of the image (and of the output).

    Returns:
        torch.Tensor: Sampling grid [B, H, W, 2].
    """
    M = F.pad(M, [0, 0, 0, 1], 'constant', value=0.0)
    M[..., -1, -1] += 1.0
    N = _pixel_to_grid(H, W, M)
    M_grid = N @ (M @ torch.inverse(N))
    return F.affine_grid(torch.inverse(M_grid)[:, :2, :], [M.shape[0], 1, H, W], align_corners=True)

def rotation_matrix_pixels(angle: float, H: int, W: int, device, dtype) -> torch.Tensor:
    """The rotation by ``angle`` degrees about the image centre, as an affine map in pixel coordinates.

    The rotation is anti-clockwise as displayed (row 0 on top), about ``((W - 1) / 2, (H - 1) / 2)``, in the pixel
    coordinates of :func:`affine_grid_pixels`. It is computed as kornia's ``get_rotation_matrix2d`` computed it
    (degrees to radians with pi in float32, then shift, rotation and shift back), so that the grids are bit-identical
    to kornia's.

    Args:
        angle (float): The angle, in degrees.
        H (int): Height of the image.
        W (int): Width of the image.
        device (torch.device | str): Device of the matrix.
        dtype (torch.dtype): Type of the matrix.

    Returns:
        torch.Tensor: The map, [1, 2, 3].
    """
    eye = torch.eye(3, device=device).type(dtype)[None]
    centre = torch.tensor([[(W - 1) / 2, (H - 1) / 2]], device=device, dtype=dtype)
    radians = torch.tensor([float(angle)], device=device, dtype=dtype) * torch.tensor(math.pi).to(device).type(dtype) / 180.0
    cos, sin = torch.cos(radians), torch.sin(radians)
    shift, rotation, unshift = eye.clone(), eye.clone(), eye.clone()
    shift[:, :2, 2] = centre
    rotation[:, :2, :2] = torch.stack([cos, sin, -sin, cos], dim=-1).view(1, 2, 2)
    unshift[:, :2, 2] = -centre
    return (shift @ rotation @ unshift)[:, :2, :]

def _rotation_grid(angle: float, H: int, W: int, device, dtype) -> torch.Tensor:
    """Sampling grid of the rotation by ``angle`` (degrees, anti-clockwise about the image centre), built once and cached. Building it per call (rotation matrix, normalisation, a 3x3 inverse on the device, ``affine_grid``) costs ~30 kernel launches per angle and makes the rotate+sum projector launch-bound: on an RTX 5090 a 64^3 forward projection over 96 angles spends 520 ms of wall time for 15 ms of GPU time."""
    key = (round(float(angle), 6), H, W, str(device), dtype)
    grid = _ROTATION_GRIDS.get(key)
    if grid is None:
        grid = affine_grid_pixels(rotation_matrix_pixels(angle, H, W, device, dtype), H, W)
        _ROTATION_GRIDS[key] = grid
    return grid

def rotate_cached(x: torch.Tensor, angle, mode: str = 'bilinear') -> torch.Tensor:
    """Rotates images [B, C, H, W] by one angle about their centre, with a sampling grid cached per angle and size.

    The rotation is anti-clockwise as displayed, with zeros outside the image: one ``grid_sample`` call per rotation.
    The result is bit-identical to ``kornia.geometry.transform.rotate(x, angle, mode=mode)``, which PyTomography used
    before 4.0.

    Args:
        x (torch.Tensor): The images, [B, C, H, W].
        angle (float | torch.Tensor): The rotation angle in degrees (a number or a 0-d tensor).
        mode (str, optional): Interpolation, ``'bilinear'`` or ``'nearest'``. Defaults to ``'bilinear'``.

    Returns:
        torch.Tensor: The rotated images, [B, C, H, W].
    """
    B, C, H, W = x.shape
    grid = _rotation_grid(float(angle), H, W, x.device, x.dtype)
    return F.grid_sample(x, grid.expand(B, -1, -1, -1), mode=mode, padding_mode='zeros', align_corners=True)

def rotate_detector_z(
    x: torch.Tensor,
    angles: torch.tensor,
    mode: str = 'bilinear',
    negative: bool = False):
    r"""Rotates an object about z into the frame of the SPECT detector at angle :math:`\beta`: the detector at +x.

    The rotation is by :math:`-\phi` with :math:`\phi = 270° - \beta` (:func:`rotate_cached` on each plane of
    constant z), as :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix` rotates an object before summing it along
    x; ``negative=True`` rotates by :math:`+\phi`, back from that frame.

    Args:
        x (torch.Tensor): The object, [Lx, Ly, Lz] (padded with :func:`pad_object`, so that nothing is cut off).
        angles (float | torch.Tensor): The detector's angle :math:`\beta`, in degrees (a number or a 0-d tensor).
        mode (str, optional): Interpolation, ``'bilinear'`` or ``'nearest'``. Defaults to ``'bilinear'``.
        negative (bool, optional): Rotate back from the detector's frame to the object's (the inverse rotation).
            Defaults to False.

    Returns:
        torch.Tensor: The rotated object, [Lx, Ly, Lz].
    """
    phi = 270 - angles
    if not negative:
        x = rotate_cached(x.permute(2,0,1).unsqueeze(0), -phi, mode=mode).squeeze().permute(1,2,0)
    else:
        x = rotate_cached(x.permute(2,0,1).unsqueeze(0), phi, mode=mode).squeeze().permute(1,2,0)
    return x

def compute_pad_size(width: int):
    """The padding on each side of an axis that keeps an object inside its grid whatever it is rotated by.

    It is :math:`\\lceil (\\sqrt{2} - 1) w / 2 \\rceil`, so that the padded width covers the diagonal.

    Args:
        width (int): Number of voxels along the axis.

    Returns:
        int: Number of voxels to add on each side.
    """
    return int(np.ceil((np.sqrt(2)*width - width)/2)) 

def compute_pad_size_padded(width: int):
    """The padding on each side of an axis padded by :func:`compute_pad_size`, from its padded width.

    Args:
        width (int): Number of voxels along the padded axis.

    Returns:
        int: Number of voxels added on each side.
    """
    # Note: This seemed to empirically work for all integers between 1 and 10million, so I'll use it
    a = (np.sqrt(2) - 1)/2
    if width%2==0:
        width_old = int(2*np.floor((width/2)/(1+2*a)))
    else:
        width_old = int(2*np.floor(((width-1)/2)/(1+2*a)))
    return int((width-width_old)/2)

def pad_object(object: torch.Tensor, mode='constant'):
    """Pads an object along x and y so that rotating it about z cuts nothing off.

    Both axes get :func:`compute_pad_size` of Ly voxels on each side.

    Args:
        object (torch.Tensor): The object, [Lx, Ly, Lz].
        mode (str, optional): A padding mode of ``torch.nn.functional.pad`` (``'constant'`` pads with zeros), or
            ``'back_project'``: the edge values repeated along x (the axis back projection spreads along) and zeros
            along y. Defaults to ``'constant'``.

    Returns:
        torch.Tensor: The padded object, [Lx + 2 p, Ly + 2 p, Lz] with p = ``compute_pad_size(Ly)``.
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
    """Removes the padding of :func:`pad_object` along x and y.

    Args:
        object (torch.Tensor): The padded object, [Lx', Ly', Lz].

    Returns:
        torch.Tensor: The object, [Lx, Ly, Lz] (a view of the padded one).
    """
    pad_size = compute_pad_size_padded(object.shape[-2])
    return object[pad_size:-pad_size,pad_size:-pad_size,:]

def pad_proj(proj: torch.Tensor, mode: str = 'constant', value: float = 0):
    """Pads projections along r, by :func:`compute_pad_size` of Lr pixels on each side, to match a padded object.

    Args:
        proj (torch.Tensor): Projections [..., Lr, Lz], e.g. [θ, Lr, Lz] or a mask [Lr, Lz].
        mode (str, optional): A padding mode of ``torch.nn.functional.pad``. Defaults to ``'constant'``.
        value (float, optional): The value of the padding with ``mode='constant'``. Defaults to 0.

    Returns:
        torch.Tensor: The padded projections, [..., Lr', Lz].
    """
    pad_size = compute_pad_size(proj.shape[-2])  
    return pad(proj, [0,0,pad_size,pad_size], mode=mode, value=value)

def unpad_proj(proj: torch.Tensor):
    """Removes the padding of :func:`pad_proj` along r.

    Args:
        proj (torch.Tensor): The padded projections, [θ, Lr', Lz].

    Returns:
        torch.Tensor: The projections, [θ, Lr, Lz] (a view of the padded ones).
    """
    pad_size = compute_pad_size_padded(proj.shape[-2])
    return proj[:,pad_size:-pad_size,:]

def pad_object_z(object: torch.Tensor, pad_size: int, mode='constant'):
    """Pads an object along z by ``pad_size`` voxels on each side.

    Args:
        object (torch.Tensor): The object, [Lx, Ly, Lz].
        pad_size (int): Number of voxels added at each end of z.
        mode (str, optional): A padding mode of ``torch.nn.functional.pad``. Defaults to ``'constant'``.

    Returns:
        torch.Tensor: The padded object, [Lx, Ly, Lz + 2 pad_size].
    """
    return pad(object, [pad_size,pad_size,0,0,0,0], mode=mode)

def unpad_object_z(object: torch.Tensor, pad_size: int):
    """Removes ``pad_size`` voxels from each end of z (the inverse of :func:`pad_object_z`).

    Args:
        object (torch.Tensor): The padded object, [Lx, Ly, Lz'].
        pad_size (int): Number of voxels that were added at each end of z.

    Returns:
        torch.Tensor: The object, [Lx, Ly, Lz' - 2 pad_size] (a view of the padded one).
    """
    
    return object[:,:,pad_size:-pad_size]
