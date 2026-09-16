import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

module_path = os.path.dirname(os.path.abspath(__file__))
color_filepath = os.path.join(module_path, "../data/pet_colors.txt")
colors = np.loadtxt(color_filepath).reshape(-1,3)/255.0
pet_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

def get_extent(im, affine):
    dx, dy, dz = affine[0,0], affine[1,1], affine[2,2]
    extent_x = [affine[0,3]-0.5*dx, affine[0,3]+(-0.5+im.shape[0])*dx]
    extent_y = [affine[1,3]-0.5*dy, affine[1,3]+(-0.5+im.shape[1])*dy]
    extent_z = [affine[2,3]-0.5*dz, affine[2,3]+(-0.5+im.shape[2])*dz]
    return extent_x, extent_y, extent_z

def dual_imshow(plane, im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    extent1_x, extent1_y, extent1_z = get_extent(im1, affine1)
    extent2_x, extent2_y, extent2_z = get_extent(im2, affine2)
    IDX_1 = im1_idx
    if plane=='coronal':
        IDX_2 = round((extent1_y[0] - extent2_y[0] + affine1[1,1] * IDX_1)/affine2[1,1])
        slice_1 = im1[:,IDX_1].cpu().T
        slice_2 = im2[:,IDX_2].cpu().T
        extent1 = [*extent1_x, *extent1_z]
        extent2 = [*extent2_x, *extent2_z]
    elif plane=='axial':
        IDX_2 = round((extent1_z[0] - extent2_z[0] + affine1[2,2] * IDX_1)/affine2[2,2])
        slice_1 = im1[:,:,IDX_1].cpu().T
        slice_2 = im2[:,:,IDX_2].cpu().T
        extent1 = [*extent1_x, *extent1_y]
        extent2 = [*extent2_x, *extent2_y]
    elif plane=='sagittal':
        IDX_2 = round((extent1_x[0] - extent2_x[0] + affine1[0,0] * IDX_1)/affine2[0,0])
        slice_1 = im1[IDX_1].cpu().T
        slice_2 = im2[IDX_2].cpu().T
        extent1 = [*extent1_y, *extent1_z]
        extent2 = [*extent2_y, *extent2_z]
    plt.imshow(slice_1, **imshow1_kwargs, extent=extent1)
    plt.imshow(slice_2, **imshow2_kwargs, extent=extent2)
    plt.xlim(extent1[0], extent1[1])
    plt.ylim(extent1[2], extent1[3])

def dual_imshow_coronal(im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    return dual_imshow('coronal', im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs)

def dual_imshow_axial(im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    return dual_imshow('axial', im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs)

def dual_imshow_sagittal(im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs):
    return dual_imshow('sagittal', im1, im2, im1_idx, affine1, affine2, imshow1_kwargs, imshow2_kwargs)