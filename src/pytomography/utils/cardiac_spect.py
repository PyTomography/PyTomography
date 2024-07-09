import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, shift
from pytomography.utils.spatial import euler_angle_transform

def get_reorientation_params(object: torch.Tensor, lowerThreshold: float, upperThreshold: float):
    """Getting the reorientation parameters for the given object.

    Args:
        object: torch.Tensor[batch_size, Lx, Ly, Lz']
        lowerThreshold: Lower threshold for the binary mask
        upperThreshold: Upper threshold for the binary mask

    Returns:
        Six degree of freedom for reorientation including translation (tx, ty, tz) and rotation (beta, alpha, gamma) parameters.
    """

    axial_slice = object[0, :, :, object.shape[3] // 2].T  # Extract the suitable axial slice data
    lower_threshold, upper_threshold = lowerThreshold * axial_slice.max(), upperThreshold * axial_slice.max()
    binary_mask = (axial_slice > lower_threshold) & (axial_slice < upper_threshold)
    binary_mask = binary_mask.float()

    com = center_of_mass(binary_mask.cpu().numpy())
    center_of_image = torch.tensor(binary_mask.shape, dtype=torch.float32) / 2
    shift_values = center_of_image - torch.tensor(com, dtype=torch.float32)
    shifted_binary_mask = shift(binary_mask.cpu().numpy().astype(float), shift=shift_values.numpy(), order=0)
    shifted_binary_mask = torch.from_numpy(shifted_binary_mask)

    y_indices, x_indices = torch.nonzero(shifted_binary_mask, as_tuple=True)
    angles_rad = torch.deg2rad(torch.linspace(-60, -25, 90))  # Define the angle range and convert to radians
    slopes = torch.tan(angles_rad)
    intercepts = center_of_image[1] - slopes * center_of_image[0]
    total_distances = torch.sum(torch.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) / torch.sqrt(slopes[:, None]**2 + 1), dim=1)
    best_idx = torch.argmin(total_distances)
    best_angle, best_slope, best_intercept = angles_rad[best_idx], slopes[best_idx], intercepts[best_idx]
    x_line = torch.linspace(0, shifted_binary_mask.shape[1], 500)
    y_line = best_slope * x_line + best_intercept
    angle_display = 90 + torch.rad2deg(best_angle)

    tx = shift_values[0] + axial_slice.shape[1] / 2
    ty = shift_values[1]
    tz = 0
    beta = 0
    alpha = 0
    gamma = angle_display.item()

    result_volume = euler_angle_transform(object, tx.item(), ty.item(), tz, beta, alpha, gamma)

    axial_slice = result_volume[0, result_volume.shape[1] // 2, :, :].T  # Extract the suitable axial slice data
    lower_threshold, upper_threshold = 0.55 * axial_slice.max(), 1 * axial_slice.max()
    binary_mask = (axial_slice > lower_threshold) & (axial_slice < upper_threshold)
    binary_mask = binary_mask.float()

    com = center_of_mass(binary_mask.cpu().numpy())
    center_of_image = torch.tensor(binary_mask.shape, dtype=torch.float32) / 2
    shift_values = center_of_image - torch.tensor(com, dtype=torch.float32)
    shifted_binary_mask = shift(binary_mask.cpu().numpy().astype(float), shift=shift_values.numpy(), order=0)
    shifted_binary_mask = torch.from_numpy(shifted_binary_mask)

    y_indices, x_indices = torch.nonzero(shifted_binary_mask, as_tuple=True)
    angles_rad = torch.deg2rad(torch.linspace(-5, 5, 90))  # Define the angle range and convert to radians
    slopes = torch.tan(angles_rad)
    intercepts = center_of_image[1] - slopes * center_of_image[0]
    total_distances = torch.sum(torch.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) / torch.sqrt(slopes[:, None]**2 + 1), dim=1)
    best_idx = torch.argmin(total_distances)
    best_angle, best_slope, best_intercept = angles_rad[best_idx], slopes[best_idx], intercepts[best_idx]
    best_intercept = com[0] - best_slope.item() * com[1]
    mirrored_mask_horizontal = torch.flip(shifted_binary_mask, [1])  # Reverse the tensor along the horizontal axis
    x_line = torch.linspace(0, mirrored_mask_horizontal.shape[1], 500)
    y_line = best_slope * x_line + best_intercept
    angle_display = -1 * torch.rad2deg(best_angle).item()

    tz = shift_values[1]
    beta = angle_display

    return tx.item(), ty.item(), tz.item(), beta, alpha, gamma

