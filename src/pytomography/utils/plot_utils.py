"""Plotting helpers: PyTomography's PET colour map, and one image drawn over another, both placed by their affines.

Importing this module (which ``import pytomography.utils`` does) registers :data:`pet_cmap` with matplotlib as
``'pet'``. :func:`dual_imshow` (or :func:`dual_imshow_axial`, :func:`dual_imshow_coronal`, :func:`dual_imshow_sagittal`)
draws a slice of one image with the matching slice of a second one over it, e.g. a reconstruction over its CT.
"""
import numpy as np
import os
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

module_path = os.path.dirname(os.path.abspath(__file__))
color_filepath = os.path.join(module_path, "../data/pet_colors.txt")
colors = np.loadtxt(color_filepath).reshape(-1,3)/255.0
#: PyTomography's PET colour scale (black, teal, violet, orange, white; ``data/pet_colors.txt``), also registered with
#: matplotlib by name: ``cmap='pet'`` works once ``pytomography.utils`` is imported (``pytomography.io`` imports it).
pet_cmap = LinearSegmentedColormap.from_list('pet', colors)
if 'pet' not in matplotlib.colormaps:
    matplotlib.colormaps.register(pet_cmap)

_AXIS = {'sagittal': 0, 'coronal': 1, 'axial': 2}


def _numpy(im) -> np.ndarray:
    if hasattr(im, 'detach'):
        im = im.detach().cpu().numpy()
    return np.asarray(im)


def _diagonal(affine) -> np.ndarray:
    """The affine as a float64 numpy array, checked to be axis-aligned (voxel sizes on the diagonal, possibly
    negative)."""
    A = np.asarray(_numpy(affine), dtype=np.float64).reshape(4, 4)
    d = np.diag(A)[:3]
    if not np.allclose(A[:3, :3], np.diag(d), atol=1e-6 * np.abs(d).max()):
        raise ValueError('dual_imshow needs axis-aligned affines (no rotations); resample the image first')
    return A


def get_extent(im, affine):
    """The extent of an image along each of its three axes, in the affine's coordinates (mm for PyTomography's affines).

    Along each axis the extent runs from the outer edge of the first voxel to the outer edge of the last one, so it
    decreases where the affine's voxel size is negative.

    Args:
        im (np.ndarray | torch.Tensor): Image [Lx, Ly, Lz] (only its shape is used).
        affine (np.ndarray | torch.Tensor): The 4 x 4 affine from voxel index to coordinates, e.g.
            ``object_meta.affine``; axis-aligned (no rotation).

    Returns:
        tuple[list[float], list[float], list[float]]: ``[first edge, last edge]`` along each of the three axes.

    Raises:
        ValueError: The affine is not axis-aligned.
    """
    A = _diagonal(affine)
    extents = []
    for axis in range(3):
        d, origin = A[axis, axis], A[axis, 3]
        extents.append([origin - 0.5 * d, origin + (-0.5 + im.shape[axis]) * d])
    return tuple(extents)


def matching_slice(im1_idx: int, affine1, affine2, axis: int, n2: int):
    """The index of the slice of image 2 whose centre is nearest the centre of slice ``im1_idx`` of image 1.

    Args:
        im1_idx (int): Index of the slice of image 1 along ``axis``.
        affine1 (np.ndarray | torch.Tensor): Axis-aligned 4 x 4 affine of image 1 (voxel index to coordinates).
        affine2 (np.ndarray | torch.Tensor): Axis-aligned 4 x 4 affine of image 2, in the same coordinates.
        axis (int): Axis across the slices (0, 1 or 2).
        n2 (int): Number of slices of image 2 along ``axis``.

    Returns:
        int | None: The index of the slice of image 2, or None when that index is outside ``[0, n2)``.

    Raises:
        ValueError: An affine is not axis-aligned.
    """
    A1, A2 = _diagonal(affine1), _diagonal(affine2)
    position = A1[axis, 3] + A1[axis, axis] * im1_idx
    idx2 = int(np.round((position - A2[axis, 3]) / A2[axis, axis]))
    return idx2 if 0 <= idx2 < n2 else None


def dual_imshow(plane, im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    """Draws slice ``im1_idx`` of image 1 with the matching slice of image 2, e.g. a reconstruction and its CT.

    Both images are placed by their affines (voxel index to patient coordinates in the same length unit, e.g.
    ``object_meta.affine`` and the ``affine`` of an :class:`~pytomography.io.Image`, both mm), which must be
    axis-aligned. The slice of image 2 is the one whose centre is nearest the centre of image 1's slice
    (:func:`matching_slice`); image 2 is not drawn when image 1's slice is outside it. Each slice is drawn on the current
    axes with ``plt.imshow``, transposed so that the first of its two remaining axes runs horizontally, with an
    ``extent`` from its first voxel's outer edge to its last's (:func:`get_extent`). Each image is placed by its affine
    whatever ``origin`` its keyword dict gives (or matplotlib's default), and the axes run along increasing coordinates:
    the axis limits are image 1's extent. Which image is on top is set by ``zorder`` in the keyword dicts.

    Args:
        plane (str): ``'sagittal'`` (slices across the first axis), ``'coronal'`` (the second) or ``'axial'`` (the
            third).
        im1 (np.ndarray | torch.Tensor): Image 1 [Lx, Ly, Lz].
        im2 (np.ndarray | torch.Tensor): Image 2, of any shape and voxel size.
        im1_idx (int): Index of the slice of image 1.
        affine1 (np.ndarray | torch.Tensor): 4 x 4 affine of image 1.
        affine2 (np.ndarray | torch.Tensor): 4 x 4 affine of image 2.
        imshow1_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 1 (e.g. ``cmap``, ``vmax``, ``origin``).
        imshow2_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 2.

    Raises:
        ValueError: An affine is not axis-aligned.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    im1, im2 = _numpy(im1), _numpy(im2)
    axis = _AXIS[plane]
    extent1 = get_extent(im1, affine1)
    extent2 = get_extent(im2, affine2)
    IDX_2 = matching_slice(im1_idx, affine1, affine2, axis, im2.shape[axis])
    shown = [a for a in range(3) if a != axis]

    def extent(extents, kwargs):
        # imshow puts a slice's first row at the bottom of the extent with origin 'lower', at the top with 'upper'
        rows = extents[shown[1]]
        if kwargs.get('origin', matplotlib.rcParams['image.origin']) == 'upper':
            rows = rows[::-1]
        return [*extents[shown[0]], *rows]

    slice_1 = np.take(im1, im1_idx, axis=axis).T
    plt.imshow(slice_1, **imshow1_kwargs, extent=extent(extent1, imshow1_kwargs))
    if IDX_2 is not None:
        slice_2 = np.take(im2, IDX_2, axis=axis).T
        plt.imshow(slice_2, **imshow2_kwargs, extent=extent(extent2, imshow2_kwargs))
    plt.xlim(*extent1[shown[0]])
    plt.ylim(*extent1[shown[1]])

def dual_imshow_coronal(im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    """Draws coronal slice ``im1_idx`` (second axis) of image 1 with the matching slice of image 2: :func:`dual_imshow`.

    Args:
        im1 (np.ndarray | torch.Tensor): Image 1 [Lx, Ly, Lz].
        im2 (np.ndarray | torch.Tensor): Image 2.
        im1_idx (int): Index of the slice of image 1 along its second axis.
        affine1 (np.ndarray | torch.Tensor): 4 x 4 affine of image 1.
        affine2 (np.ndarray | torch.Tensor): 4 x 4 affine of image 2.
        imshow1_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 1.
        imshow2_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 2.
    """
    return dual_imshow('coronal', im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs)

def dual_imshow_axial(im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    """Draws axial slice ``im1_idx`` (third axis) of image 1 with the matching slice of image 2: :func:`dual_imshow`.

    Args:
        im1 (np.ndarray | torch.Tensor): Image 1 [Lx, Ly, Lz].
        im2 (np.ndarray | torch.Tensor): Image 2.
        im1_idx (int): Index of the slice of image 1 along its third axis.
        affine1 (np.ndarray | torch.Tensor): 4 x 4 affine of image 1.
        affine2 (np.ndarray | torch.Tensor): 4 x 4 affine of image 2.
        imshow1_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 1.
        imshow2_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 2.
    """
    return dual_imshow('axial', im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs)

def dual_imshow_sagittal(im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    """Draws sagittal slice ``im1_idx`` (first axis) of image 1 with the matching slice of image 2: :func:`dual_imshow`.

    Args:
        im1 (np.ndarray | torch.Tensor): Image 1 [Lx, Ly, Lz].
        im2 (np.ndarray | torch.Tensor): Image 2.
        im1_idx (int): Index of the slice of image 1 along its first axis.
        affine1 (np.ndarray | torch.Tensor): 4 x 4 affine of image 1.
        affine2 (np.ndarray | torch.Tensor): 4 x 4 affine of image 2.
        imshow1_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 1.
        imshow2_kwargs (dict): Keyword arguments of ``plt.imshow`` for image 2.
    """
    return dual_imshow('sagittal', im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs)
