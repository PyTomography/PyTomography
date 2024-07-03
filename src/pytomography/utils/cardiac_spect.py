import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from scipy.ndimage import center_of_mass, shift
import torch

def plot_slices(volume, title, batch_idx=0):
    print(f"Volume shape: {volume.shape}")  # Debugging print
    batch_volume = volume[batch_idx]
    print(f"Batch volume shape: {batch_volume.shape}")  # Debugging print
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        np.flip(batch_volume[batch_volume.shape[0] // 2, :, :].T, axis=1),  # Mirrored Sagittal
        # batch_volume[batch_volume.shape[0] // 2, :, :].T,  # Mirrored Sagittal
        batch_volume[:, batch_volume.shape[1] // 2, :].T,   # Coronal
        batch_volume[:, :, batch_volume.shape[2] // 2].T    # Axial
    ]
    
    for ax, slice_img, label in zip(axes, slices, ['HLA', 'SA', 'VLA']):
        print(f"{label} slice shape: {slice_img.shape}")  # Debugging print
        ax.imshow(slice_img, cmap='jet')
        ax.set_title(label)
        ax.axis('on')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def object_reorient(volume, tx, ty, tz, beta, alpha, gamma):
    """
    Reorient the 3D volume based on the given translation and rotation parameters.
    
    Parameters:
        volume (numpy.ndarray): Input 3D volume.
        tx (float): Translation in x direction (voxels).
        ty (float): Translation in y direction (voxels).
        tz (float): Translation in z direction (voxels).
        beta (float): Rotation around x-axis (radians).
        alpha (float): Rotation around y-axis (radians).
        gamma (float): Rotation around z-axis (radians).
    
    Returns:
        numpy.ndarray: Reoriented 3D volume.
    """
    volume = volume.squeeze() 
    
    beta, alpha, gamma = np.deg2rad(beta), np.deg2rad(alpha), np.deg2rad(gamma)  # Example rotation values in radians
    
    translation = [tx, ty, tz]
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(beta), -np.sin(beta)],
                   [0, np.sin(beta), np.cos(beta)]])
    
    Ry = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                   [0, 1, 0],
                   [-np.sin(alpha), 0, np.cos(alpha)]])
    
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    
    R = Rz@Ry@Rx
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = translation
    
    affine_matrix = transform_matrix[:3, :4]
    
    reoriented_volume = affine_transform(volume, affine_matrix, offset=translation, order=3)
    
    # Add a batch dimension
    reoriented_volume = np.expand_dims(reoriented_volume, axis=0)
    
    return reoriented_volume
  
  

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
