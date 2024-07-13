import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift
from pytomography.utils.spatial import euler_angle_transform

def create_binary_mask(object: torch.Tensor, lowerThreshold: float, upperThreshold: float):
    """Segment the object based on the given thresholds.
    
    Args:
        volume: torch.Tensor[Lx, Ly, Lz]
        lowerThreshold: Lower threshold for the binary mask
        upperThreshold: Upper threshold for the binary mask
    """
    
    lower_threshold, upper_threshold = lowerThreshold * object.max(), upperThreshold * object.max()
    mask = (object > lower_threshold) & (object < upper_threshold)
    
    return mask.float()

def plot_slice_alignment(shifted_binary_mask: torch.Tensor, x_line: torch.Tensor, y_line: torch.Tensor, center_of_image: torch.Tensor, angle_display: float):
    """Show the selected angle alignment on the binary mask.

    Args:
        shifted_binary_mask: torch.Tensor representing the binary mask of the slice.
        x_line: torch.Tensor representing the x-coordinates of the fitted line.
        y_line: torch.Tensor representing the y-coordinates of the fitted line.
        center_of_image: torch.Tensor representing the center of the image.
        angle_display: float representing the angle to display in the plot title.
    """
    
    plt.imshow(shifted_binary_mask, cmap='gray')
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2)
    plt.scatter(center_of_image[1], center_of_image[0], color='yellow', marker='x', label='Center of Mass')
    plt.title(f"Shifted Binary Mask with Fitted Line at {angle_display:.1f}°")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def get_reorientation_params(object: torch.Tensor, lowerThreshold: float, upperThreshold: float, axial_slice=None, axial_angle=None):
    """Getting the reorientation parameters for the given object which can be either an object or segmentation of the object.

    Args:
        object: torch.Tensor[Lx, Ly, Lz]
        axial_slice: Precomputed axial slice if available
        axial_angles: Precomputed angles in radians for first reorientation if available

    Returns:
        Six degree of freedom for reorientation including translation (tx, ty, tz) and rotation (beta, alpha, gamma) parameters.
    """
    
    axial_slice = object[:, :, object.shape[2] // 2].T if axial_slice is None else axial_slice
    axial_slice = create_binary_mask(axial_slice, lowerThreshold, upperThreshold)
    
    com = center_of_mass(axial_slice.cpu().numpy())
    center_of_image = torch.tensor(axial_slice.shape, dtype=torch.float32) / 2
    shift_values = center_of_image - torch.tensor(com, dtype=torch.float32)
    shifted_binary_mask = shift(axial_slice.cpu().numpy().astype(float), shift=shift_values.numpy(), order=0)
    shifted_binary_mask = torch.from_numpy(shifted_binary_mask)
    y_indices, x_indices = torch.nonzero(shifted_binary_mask, as_tuple=True)
    
    axial_angle = torch.deg2rad(torch.linspace(-60, -25, 90)) if axial_angle is None else axial_angle  # Define the angle range and convert to radians
    slopes = torch.tan(axial_angle)
    intercepts = center_of_image[1] - slopes * center_of_image[0]
    total_distances = torch.sum(torch.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) / torch.sqrt(slopes[:, None]**2 + 1), dim=1)
    best_idx = torch.argmin(total_distances)
    best_angle, best_slope, best_intercept = axial_angle[best_idx], slopes[best_idx], intercepts[best_idx]
    x_line = torch.linspace(0, shifted_binary_mask.shape[1], 500)
    y_line = best_slope * x_line + best_intercept
    angle_display = 90 + torch.rad2deg(best_angle)
    plot_slice_alignment(shifted_binary_mask, x_line, y_line, center_of_image, angle_display)
    
    tx, ty, tz = shift_values[0] + axial_slice.shape[1] / 2, shift_values[1], 0
    beta, alpha, gamma = 0, 0, angle_display.item()

    result_volume = euler_angle_transform(object, tx.item(), ty.item(), tz, beta, alpha, gamma)
    HLA_slice = result_volume[result_volume.shape[0] // 2, :, :].T
    HLA_slice = create_binary_mask(HLA_slice, lowerThreshold, upperThreshold)
    
    com = center_of_mass(HLA_slice.cpu().numpy())
    center_of_image = torch.tensor(HLA_slice.shape, dtype=torch.float32) / 2
    shift_values = center_of_image - torch.tensor(com, dtype=torch.float32)
    shifted_binary_mask = shift(HLA_slice.cpu().numpy().astype(float), shift=shift_values.numpy(), order=0)
    shifted_binary_mask = torch.from_numpy(shifted_binary_mask)
    y_indices, x_indices = torch.nonzero(shifted_binary_mask, as_tuple=True)
    
    HLA_angle = torch.deg2rad(torch.linspace(-5, 5, 90))
    slopes = torch.tan(HLA_angle)
    intercepts = center_of_image[1] - slopes * center_of_image[0]
    total_distances = torch.sum(torch.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) / torch.sqrt(slopes[:, None]**2 + 1), dim=1)
    best_idx = torch.argmin(total_distances)
    best_angle, best_slope, best_intercept = HLA_angle[best_idx], slopes[best_idx], intercepts[best_idx]
    best_intercept = com[0] - best_slope.item() * com[1]
    mirrored_mask_horizontal = torch.flip(shifted_binary_mask, [1])  # Reverse the tensor along the horizontal axis
    x_line = torch.linspace(0, mirrored_mask_horizontal.shape[1], 500)
    y_line = best_slope * x_line + best_intercept
    angle_display = -1 * torch.rad2deg(best_angle).item()
    plot_slice_alignment(shifted_binary_mask, x_line, y_line, center_of_image, angle_display)
    
    tz, beta = shift_values[1], angle_display

    return tx.item(), ty.item(), tz.item(), beta, alpha, gamma
