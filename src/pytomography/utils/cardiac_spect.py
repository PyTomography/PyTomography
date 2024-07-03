import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.ndimage import center_of_mass, shift
import torch
  
  
def reorientParams(volume, lowerThreshold=0.55, upperThreshold=1):
    volume = volume.numpy() if isinstance(volume, torch.Tensor) else volume  # Ensure volume is a NumPy array
    axial_slice = volume[0, :, :, volume.shape[3] // 2].T  # Extract the suitable axial slice data
    lower_threshold, upper_threshold = lowerThreshold * axial_slice.max(), upperThreshold * axial_slice.max()
    binary_mask = (axial_slice > lower_threshold) & (axial_slice < upper_threshold)
    com = center_of_mass(binary_mask)
    center_of_image = np.array(binary_mask.shape) / 2
    shift_values = center_of_image - np.array(com)
    shifted_binary_mask = shift(binary_mask.astype(float), shift=shift_values, order=0)

    y_indices, x_indices = np.nonzero(shifted_binary_mask)
    angles_rad = np.deg2rad(np.linspace(-60, -25, 100)) # Define the angle range and convert to radians
    slopes = np.tan(angles_rad)
    intercepts = center_of_image[1] - slopes * center_of_image[0]
    total_distances = np.sum(np.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) / np.sqrt(slopes[:, None]**2 + 1), axis=1)
    best_idx = np.argmin(total_distances)
    best_angle, best_slope, best_intercept = angles_rad[best_idx], slopes[best_idx], intercepts[best_idx]
    x_line = np.linspace(0, shifted_binary_mask.shape[1], 500)
    y_line = best_slope * x_line + best_intercept
    angle_display = 90 + np.rad2deg(best_angle)

    plt.imshow(shifted_binary_mask, cmap='gray')
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2)
    plt.scatter(center_of_image[1], center_of_image[0], color='yellow', marker='x', label='Center of Mass')
    plt.title(f"Shifted Binary Mask with Fitted Line at {angle_display:.1f}°")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    tx = shift_values[0] + axial_slice.shape[1] / 2
    ty = shift_values[1]
    tz = 0
    beta = 0
    alpha = 0
    gamma = angle_display
    
    result_volume = object_reorient(volume, tx, ty, tz, beta, alpha, gamma)
    plot_slices(result_volume, 'Reoriented Volume')
    
    axial_slice = result_volume[0, result_volume.shape[1] // 2, :, :].T  # Extract the suitable axial slice data
    lower_threshold, upper_threshold = 0.55 * axial_slice.max(), 1 * axial_slice.max()
    binary_mask = (axial_slice > lower_threshold) & (axial_slice < upper_threshold)
    com = center_of_mass(binary_mask)
    center_of_image = np.array(binary_mask.shape) / 2
    shift_values = center_of_image - np.array(com)
    shifted_binary_mask = shift(binary_mask.astype(float), shift=shift_values, order=0)

    y_indices, x_indices = np.nonzero(shifted_binary_mask)
    angles_rad = np.deg2rad(np.linspace(-5, 5, 100)) # Define the angle range and convert to radians
    slopes = np.tan(angles_rad)
    intercepts = center_of_image[1] - slopes * center_of_image[0]
    total_distances = np.sum(np.abs(slopes[:, None] * x_indices - y_indices + intercepts[:, None]) / np.sqrt(slopes[:, None]**2 + 1), axis=1)
    best_idx = np.argmin(total_distances)
    best_angle, best_slope, best_intercept = angles_rad[best_idx], slopes[best_idx], intercepts[best_idx]
    best_intercept = com[0] - best_slope * com[1]
    mirrored_mask_horizontal = shifted_binary_mask[:, ::-1]
    x_line = np.linspace(0, mirrored_mask_horizontal.shape[1], 500)
    y_line = best_slope * x_line + best_intercept
    angle_display = -1 * np.rad2deg(best_angle)
    plt.imshow(mirrored_mask_horizontal, cmap='gray')
    plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2)
    plt.scatter(center_of_image[1], center_of_image[0], color='yellow', marker='x', label='Center of Mass')
    plt.title(f"Shifted Binary Mask with Fitted Line at {angle_display:.1f}°")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    tz = shift_values[1]
    beta = angle_display
    
    result_volume = object_reorient(volume, tx, ty, tz, beta, alpha, gamma)
    plot_slices(result_volume, 'Reoriented Volume')
    
    print(f"Translation (tx, ty, tz): ({tx}, {ty}, {tz})")
    print(f"Rotation (beta, alpha, gamma): ({beta}, {alpha}, {gamma})")
    return result_volume, tx, ty, tz, beta, alpha, gamma
