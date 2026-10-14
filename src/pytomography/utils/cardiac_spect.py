"""Helpers to reorient a cardiac SPECT image along the axes of the heart.

The steps, as in the cardiac reorientation tutorial: :func:`get_mask` thresholds the myocardium,
:func:`get_shift_values` and :func:`get_angle` find the centre and the orientation of the ventricle in a slice of the
mask, :func:`shift_object` and :func:`rotate_object` move the image accordingly, and :func:`masking` keeps the voxels
near the centre. :func:`plot_arrow` shows an angle on a slice. Shifts are in voxels and angles in degrees.
"""
import math
import torch
import torch.nn.functional as F
import pytomography
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift, affine_transform
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import rotate

def shift_object(object: torch.Tensor, tx: float, ty: float, tz: float) -> torch.Tensor:
    """Shifts a 3D image by whole or fractional voxels, with linear interpolation.

    The value of the result at voxel :math:`(i, j, k)` is that of ``object`` at :math:`(i + t_x, j + t_y, k + t_z)` (0
    outside it), so the content moves by :math:`(-t_x, -t_y, -t_z)` voxels along the three axes. Before
    PyTomography 4.0, ``tx`` and ``ty`` were applied the other way round.

    Args:
        object (torch.Tensor): Image [Lx, Ly, Lz] (a NumPy array also works).
        tx (float): Offset along the first axis (x), in voxels.
        ty (float): Offset along the second axis (y), in voxels.
        tz (float): Offset along the third axis (z), in voxels.

    Returns:
        torch.Tensor: The shifted image [Lx, Ly, Lz], of ``pytomography.dtype`` on ``pytomography.device``.
    """

    object = object.cpu().numpy() if isinstance(object, torch.Tensor) else object
    translation = [tx, ty, tz]   # x and y were swapped before PyTomography 4.0
    identity_matrix = np.eye(3)
    shifted_object = affine_transform(object, identity_matrix, offset=translation, order=1)
    
    return torch.tensor(shifted_object).to(pytomography.dtype).to(pytomography.device)

def rotate_object(object: torch.Tensor, transaxial_angle: float, transsagittal_angle: float) -> torch.Tensor:
    """Rotates a 3D image about its third axis and then about its second, keeping its shape.

    ``scipy.ndimage.rotate`` turns the image by ``transaxial_angle`` in the plane of its first two axes (about z), then
    by ``transsagittal_angle`` in the plane of its first and third axes (about y), each time about the centre of the
    array, with cubic spline interpolation and 0 outside the image.

    Args:
        object (torch.Tensor): Image [Lx, Ly, Lz].
        transaxial_angle (float): Rotation about the third axis (z), in degrees.
        transsagittal_angle (float): Rotation about the second axis (y), in degrees.

    Returns:
        torch.Tensor: The rotated image [Lx, Ly, Lz], of ``pytomography.dtype`` on ``pytomography.device``.
    """
    
    rotated_object = rotate(object.cpu(), transaxial_angle, axes=(1, 0), reshape=False)
    rotated_object = rotate(rotated_object, transsagittal_angle, axes=(2, 0), reshape=False)

    return torch.tensor(rotated_object).to(pytomography.dtype).to(pytomography.device)

def get_mask(object: torch.Tensor, lowerThreshold: float, upperThreshold: float) -> torch.Tensor:
    """Mask of the voxels whose value lies strictly between two fractions of the image's maximum.

    Args:
        object (torch.Tensor): Image [Lx, Ly, Lz].
        lowerThreshold (float): Lower threshold, as a fraction of ``object.max()``.
        upperThreshold (float): Upper threshold, as a fraction of ``object.max()``.

    Returns:
        torch.Tensor: The mask [Lx, Ly, Lz]: 1 where ``lowerThreshold * max < object < upperThreshold * max`` and 0
        elsewhere, of ``pytomography.dtype`` on ``pytomography.device``.
    """
    
    lower_threshold, upper_threshold = lowerThreshold * object.max(), upperThreshold * object.max()
    mask = (object > lower_threshold) & (object < upper_threshold)
    
    return torch.tensor(mask).to(pytomography.dtype).to(pytomography.device)

def get_shift_values(slice: torch.Tensor) -> torch.Tensor:
    """Shifts that move the centre of mass of a 2D slice to the centre of the slice, and the slice so shifted.

    For a slice of shape :math:`(N_0, N_1)` the centre is the index :math:`(N_0/2, N_1/2)`; the shifts are the centre
    minus the centre of mass, and the slice is shifted by them with nearest-neighbour interpolation (0 outside).

    Args:
        slice (torch.Tensor): Slice [N0, N1], e.g. a slice of a mask from :func:`get_mask`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The shifted slice [N0, N1] (``pytomography.dtype``, on
        ``pytomography.device``), then the shifts along its first axis (rows) and its second axis (columns), in voxels,
        as 0-dimensional float32 tensors on the CPU.
    """
    
    com = center_of_mass(slice.cpu().numpy())
    center_of_image = torch.tensor(slice.shape, dtype=torch.float32) / 2
    shift_values = center_of_image - torch.tensor(com, dtype=torch.float32)
    shifted_image = shift(slice.cpu().numpy().astype(float), shift=shift_values.cpu().numpy(), order=0)
    # shifted_image = torch.from_numpy(shifted_image)
    shifted_image = torch.tensor(shifted_image).to(pytomography.dtype).to(pytomography.device)
    
    return shifted_image, shift_values[0], shift_values[1]

def get_angle(slice: torch.Tensor, angle_range_degrees: torch.Tensor) -> float:
    r"""Angle of the line through the centre of a 2D slice that lies closest to the slice's nonzero pixels.

    The second axis (columns) is taken as x and the first (rows) as y. For each candidate angle :math:`\theta`, the line
    passes through the pixel ``(N0 // 2, N1 // 2)`` with slope :math:`\tan\theta`; the angle whose line has the smallest
    sum of perpendicular distances to the nonzero pixels is returned. Each nonzero pixel counts once, whatever its
    value.

    Args:
        slice (torch.Tensor): Slice [N0, N1], e.g. a shifted mask from :func:`get_shift_values`.
        angle_range_degrees (torch.Tensor): Candidate angles [N], in degrees, on the device of ``slice``.

    Returns:
        float: The best of the candidate angles, in degrees.
    """
    
    angle_range_radians = angle_range_degrees * (math.pi / 180)
    center_x, center_y = slice.shape[1] // 2, slice.shape[0] // 2
    y_indices, x_indices = torch.nonzero(slice, as_tuple=True)
    
    slopes = torch.tan(angle_range_radians)
    intercepts = center_y - slopes * center_x
    total_distances = torch.sum(
        torch.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) /
        torch.sqrt(slopes[:, None] ** 2 + 1), dim=1
    )
    best_idx = torch.argmin(total_distances)
    best_angle_radians = angle_range_radians[best_idx]
    best_angle_degree = math.degrees(best_angle_radians)
    
    return best_angle_degree

def plot_arrow(image, angle_degrees, fontsize=10):
    """Shows a 2D slice with an arrow from its centre at a given angle, inside a dial of angles.

    The slice is drawn in grey with its first axis running down the screen and its second to the right. The arrow's
    angle is measured clockwise from straight up; a dashed circle labelled every 10 degrees surrounds it, and the angle
    is written at the centre. The figure is shown with ``plt.show()``.

    Args:
        image (torch.Tensor): Slice [N0, N1].
        angle_degrees (float): Angle of the arrow, in degrees clockwise from up.
        fontsize (int, optional): Font size of the labels. Defaults to 10.
    """
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    angle_radians = np.deg2rad(angle_degrees)
    length = min(image.shape) // 2  # Ensure the arrow length is within the image bounds
    
    end_x = center_x + length * np.sin(angle_radians)
    end_y = center_y - length * np.cos(angle_radians)  # Note the subtraction for y
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.cpu().numpy(), cmap='gray')
    ax.add_patch(FancyArrowPatch(
        (center_x, center_y),
        (end_x, end_y),
        color='red', arrowstyle='-|>', mutation_scale=15
    ))

    num_degrees = 360
    circle = plt.Circle((center_x, center_y), length, color='red', fill=False, linestyle='--')
    ax.add_artist(circle)

    for angle in range(0, num_degrees, 10):
        angle_rad = np.deg2rad(angle)
        x = center_x + (length + 10) * np.sin(angle_rad)
        y = center_y - (length + 10) * np.cos(angle_rad)
        ax.text(x, y, f'{angle}', fontsize=fontsize, color='red', ha='center', va='center')

    ax.text(center_x, center_y, f'{angle_degrees:.2f}°', fontsize=fontsize, color='red', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))
    ax.axis('off')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0) 
    ax.set_aspect('equal')
    plt.show()
    
def create_circular_mask(height, width, radius):
    """Mask of the pixels of a 2D grid that lie within ``radius`` of its centre.

    Args:
        height (int): Number of rows (first axis).
        width (int): Number of columns (second axis).
        radius (float): Radius of the disc, in pixels.

    Returns:
        torch.Tensor: The mask [height, width], boolean: True where the distance to the pixel
        ``(height // 2, width // 2)`` is at most ``radius``.
    """
    
    Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    center_y, center_x = height // 2, width // 2
    dist_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius
    return mask

def masking(object: torch.Tensor, radius: int) -> torch.Tensor:
    """Sets to 0 the voxels of a 3D image outside three cylinders about its centre, one along each axis.

    For each axis, the disc of :func:`create_circular_mask` is applied to every slice across that axis, so the voxels
    kept are those inside all three cylinders. ``object`` itself is not changed.

    Args:
        object (torch.Tensor): Image [L0, L1, L2].
        radius (int): Radius of the cylinders, in voxels.

    Returns:
        torch.Tensor: The masked image [L0, L1, L2], of ``pytomography.dtype`` on ``pytomography.device``.
    """
    
    depth, height, width = object.shape
    mask_axial = create_circular_mask(height, width, radius)
    mask_coronal = create_circular_mask(depth, width, radius)
    mask_sagittal = create_circular_mask(depth, height, radius)
    
    masked_object = object.clone()
    for i in range(depth):
        masked_object[i, ~mask_axial] = 0
    for j in range(height):
        masked_object[:, j, :][~mask_coronal] = 0
    for k in range(width):
        masked_object[:, :, k][~mask_sagittal] = 0
    
    return torch.tensor(masked_object).to(pytomography.dtype).to(pytomography.device)