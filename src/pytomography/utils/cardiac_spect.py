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
    """
    Shift the 3D volume based on the given translation parameters.

    Parameters:
        object: Input 3D volume to be shifted (torch.Tensor[Lx, Ly, Lz]).
        tx (float): Translation in x direction (voxels).
        ty (float): Translation in y direction (voxels).
        tz (float): Translation in z direction (voxels).
    Returns:
        Shifted 3D volume: The shifted object (torch.Tensor[Lx, Ly, Lz]).
    """
    
    object = object.cpu().numpy() if isinstance(object, torch.Tensor) else object 
    translation = [ty, tx, tz]
    identity_matrix = np.eye(3)
    shifted_object = affine_transform(object, identity_matrix, offset=translation, order=1)
    
    return torch.tensor(shifted_object).to(pytomography.dtype).to(pytomography.device)

def rotate_object(object: torch.Tensor, transaxial_angle: float, transsagittal_angle: float) -> torch.Tensor:
    """
    Rotates the volume around the transaxial and transsagittal slices.

    Parameters:
        object: The 3D volume to be rotated (torch.Tensor[Lx, Ly, Lz]).
        angle_transaxial (float): The angle to rotate around the transaxial slice (z-axis).
        angle_transsagittal (float): The angle to rotate around the transsagittal slice (y-axis).

    Returns:
        The reoriented object (torch.Tensor[Lx, Ly, Lz]).
    """
    
    rotated_object = rotate(object.cpu(), transaxial_angle, axes=(1, 0), reshape=False)
    rotated_object = rotate(rotated_object, transsagittal_angle, axes=(2, 0), reshape=False)

    return torch.tensor(rotated_object).to(pytomography.dtype).to(pytomography.device)

def get_mask(object: torch.Tensor, lowerThreshold: float, upperThreshold: float) -> torch.Tensor:
    """
    Segment the object based on the given thresholds.
    
    Args:
        object: torch.Tensor[Lx, Ly, Lz]
        lowerThreshold: Lower threshold for the binary mask
        upperThreshold: Upper threshold for the binary mask
    """
    
    lower_threshold, upper_threshold = lowerThreshold * object.max(), upperThreshold * object.max()
    mask = (object > lower_threshold) & (object < upper_threshold)
    
    return torch.tensor(mask).to(pytomography.dtype).to(pytomography.device)

def get_shift_values(slice: torch.Tensor) -> torch.Tensor:
    """
    Calculate the shift values for the given object based on the center of mass.

    Args:
        slice (torch.Tensor): torch.Tensor[Lx, Ly] representing the slice.

    Returns:
        shifted_image (torch.Tensor): The shifted image.
        shift_values (torch.Tensor): The shift values in x and y directions.
    """
    
    com = center_of_mass(slice.cpu().numpy())
    center_of_image = torch.tensor(slice.shape, dtype=torch.float32) / 2
    shift_values = center_of_image - torch.tensor(com, dtype=torch.float32)
    shifted_image = shift(slice.cpu().numpy().astype(float), shift=shift_values.cpu().numpy(), order=0)
    # shifted_image = torch.from_numpy(shifted_image)
    shifted_image = torch.tensor(shifted_image).to(pytomography.dtype).to(pytomography.device)
    
    return shifted_image, shift_values[0], shift_values[1]

def get_angle(slice: torch.Tensor, angle_range_degrees: torch.Tensor) -> float:
    """
    Get the angle of the best line passing through the center of the image.

    Args:
        slice (torch.Tensor): torch.Tensor[Lx, Ly] representing the slice.
        angle_range_degrees (torch.Tensor): torch.Tensor[N] with angles in degrees.

    Returns:
        best_angle_degree: The best angle in degrees.
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
    """
    Draws an arrow from the center of the image based on the given angle and displays a polar plot around it.
    The angle is also displayed inside the plot.
    
    Parameters:
    image (np.ndarray): 2D numpy array representing the image.
    angle_degrees (float): Angle of the arrow with respect to the vertical, measured clockwise.
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
    """
    Create a circular mask.
    """
    
    Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    center_y, center_x = height // 2, width // 2
    dist_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist_from_center <= radius
    return mask

def masking(object: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Apply a circular mask to the object in all three views.
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