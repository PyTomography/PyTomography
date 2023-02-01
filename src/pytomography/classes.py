import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
import os
from pathlib import Path


# cumulative sum, but initial voxel only contriubtes 1/2 mu dx
# x: [batch_size, Lx, Ly, Lz]
def rev_cumsum(x):
    return torch.cumsum(x.flip(dims=(1,)), dim=1).flip(dims=(1,)) - x/2
    #return torch.cumsum(x, dim=1) - x/2

# Rotates the scanner scanner around an object of
# [batch_size, Lx, Ly, Lz] by angle theta in object space
# about the z axis. This is a bit tricky to understand.
# angle = beta. Rotating detector beta corresponds to rotating
# patient by -phi where phi = 3pi/2 - beta. Inverse rotatation 
# is rotating by phi (needed for back proijection)
def rotate_detector_z(x, angle, interpolation = InterpolationMode.BILINEAR, negative=False):
    phi = 270 - angle
    if not negative:
        return rotate(x.permute(0,3,1,2), -phi, interpolation).permute(0,2,3,1)
    else:
        return rotate(x.permute(0,3,1,2), phi, interpolation).permute(0,2,3,1)

# CT of size [batch_size, Lx, Ly, Lz]
def get_prob_of_detection_matrix(CT, dx): 
    return torch.exp(-rev_cumsum(CT* dx))

def get_distance(N, r, dx):
    if N%2==0:
        d = r + (N//2 - np.arange(N)) * dx
    else:
        d = r + (N//2 - np.arange(N) - 1/2) * dx
    # Correction for if radius of scanner is inside the the bounds
    d[d<0] = 0
    return d

def get_PSF_transform(sigma, kernel_size, delta=1e-9, device='cpu', kernel_dimensions='2D'):
    N = len(sigma)
    layer = torch.nn.Conv2d(N, N, kernel_size, groups=N, padding='same',
                            padding_mode='replicate', bias=0, device=device)
    x_grid, y_grid = torch.meshgrid(2*[torch.arange(-int(kernel_size//2), int(kernel_size//2)+1)],
                                    indexing='ij')
    x_grid = x_grid.unsqueeze(dim=0).repeat((N,1,1))
    y_grid = y_grid.unsqueeze(dim=0).repeat((N,1,1))
    sigma = torch.tensor(sigma, dtype=torch.float32).reshape((N,1,1))
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2*sigma**2 + delta))
    if kernel_dimensions=='1D':
        kernel[y_grid!=0] = 0
    kernel = kernel / kernel.sum(axis=(1,2)).reshape(N,1,1)
    layer.weight.data = kernel.unsqueeze(dim=1).to(device)
    return layer

def find_first_entry_containing_substring(list_of_attributes, substring, dtype=np.float32):
    line = list_of_attributes[np.char.find(list_of_attributes, substring)>=0][0]
    if dtype == np.float32:
        return np.float32(line.replace('\n', '').split(':=')[-1])
    elif dtype == str:
        return (line.replace('\n', '').split(':=')[-1].replace(' ', ''))
    elif dtype == int:
        return int(line.replace('\n', '').split(':=')[-1].replace(' ', ''))

def simind_projections_to_data(headerfile):    
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    num_proj = find_first_entry_containing_substring(headerdata, 'total number of images', int)
    proj_dim1 = find_first_entry_containing_substring(headerdata, 'matrix size [1]', int)
    proj_dim2 = find_first_entry_containing_substring(headerdata, 'matrix size [2]', int)
    dx = find_first_entry_containing_substring(headerdata, 'scaling factor (mm/pixel) [1]', np.float32) / 10 # to mm
    number_format = find_first_entry_containing_substring(headerdata, 'number format', str)
    num_bytes_per_pixel = find_first_entry_containing_substring(headerdata, 'number of bytes per pixel', np.float32)
    extent_of_rotation = find_first_entry_containing_substring(headerdata, 'extent of rotation', np.float32)
    number_of_projections = find_first_entry_containing_substring(headerdata, 'number of projections', int)
    start_angle = find_first_entry_containing_substring(headerdata, 'start angle', np.float32)
    angles = np.linspace(start_angle, extent_of_rotation, number_of_projections, endpoint=False)
    radius = find_first_entry_containing_substring(headerdata, 'Radius', np.float32) / 10
    imagefile = find_first_entry_containing_substring(headerdata, 'name of data file', str)
    shape_proj= (num_proj, proj_dim1, proj_dim2)
    shape_obj = (proj_dim1, proj_dim1, proj_dim2)
    object_meta = ObjectMeta(dx, shape_obj)
    image_meta = ImageMeta(object_meta, angles, np.ones(len(angles))*radius)
    projections = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    projections = np.transpose(projections.reshape((num_proj,proj_dim2,proj_dim1))[:,::-1], (0,2,1))
    projections = torch.tensor(projections.copy()).unsqueeze(dim=0)
    return object_meta, image_meta, projections

def simind_CT_to_data(headerfile):    
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    matrix_size_1 = find_first_entry_containing_substring(headerdata, 'matrix size [1]', int)
    matrix_size_2 = find_first_entry_containing_substring(headerdata, 'matrix size [2]', int)
    matrix_size_3 = find_first_entry_containing_substring(headerdata, 'matrix size [3]', int)
    shape = (matrix_size_3, matrix_size_2, matrix_size_1)
    imagefile = find_first_entry_containing_substring(headerdata, 'name of data file', str)
    CT = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    CT = np.transpose(CT.reshape(shape)[::-1,::-1], (2,1,0))
    CT = torch.tensor(CT.copy())
    return CT

def get_osem_net(projections_header, object_initial='ones', CT_header=None, PSF_options=None, device='cpu'):
    object_meta, image_meta, projections = simind_projections_to_data(projections_header)
    object_correction_nets = []
    image_correction_nets = []
    if CT_header:
        CT = simind_CT_to_data(CT_header)
        CT_net = CTCorrectionNet(object_meta, image_meta, CT.unsqueeze(dim=0).to(device), device=device)
        object_correction_nets.append(CT_net)
        # fill this in later
    if PSF_options:
        psf_net = PSFCorrectionNet(object_meta, image_meta, PSF_options['collimator_slope'], PSF_options['collimator_intercept'],
                           kernel_size=61, kernel_dimensions = PSF_options['kernel_dimensions'], device=device)
        object_correction_nets.append(psf_net)
        # fill this in later
    fp_net = ForwardProjectionNet(object_correction_nets, image_correction_nets,
                                object_meta, image_meta, device=device)
    bp_net = BackProjectionNet(object_correction_nets, image_correction_nets,
                                object_meta, image_meta, device=device)
    if object_initial == 'ones':
        object_initial_array = torch.ones(object_meta.shape).unsqueeze(dim=0).to(device)
    osem_net = OSEMNet(object_initial_array, fp_net, bp_net)
    osem_net.set_image(projections.to(device))
    return osem_net


class ObjectMeta():
    def __init__(self, dx, shape):
        self.dx = dx
        self.shape = shape

class ImageMeta():
    def __init__(self, object_meta, angles, radii=None):
        self.object_meta = object_meta
        self.angles = angles
        self.radii = radii
        self.num_projections = len(angles)
        self.shape = (self.num_projections, object_meta.shape[1], object_meta.shape[2])
        

# TODO: Option to store all the matrices in memory (120*128*128*128 * 32bytes = 8 Gbytes)

class CTCorrectionNet(nn.Module):
    '''Apply attenuation correction to an object.
    
    Given projection along the x-axis, an object is modified
    using an array of CT values such that subsequent projection
    yields an appropriate image
    '''
    def __init__(self, object_meta, image_meta, CT, store_in_memory=False, device='cpu'):
        '''
        Parameters
        --------
        object_meta : ObjectMeta
          Metadata for the object 
        image_meta : ImageMeta
          Metadata for the corresponding image
        CT : (batch_size, Lx, Ly, Lz) torch.tensor
          Attenuation coefficient in cm^-1 corresponding to the photon energy of the object emission data
        device : str
          The device used by pytorch where the network is placed.
        '''
        super(CTCorrectionNet, self).__init__()
        self.CT = CT
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.device = device
        self.store_in_memory = store_in_memory
        if self.store_in_memory:
            self.probability_matrices = []
            for i, angle in enumerate(self.image_meta.angles):
                self.probability_matrices.append(get_prob_of_detection_matrix(rotate_detector_z(self.CT, angle), self.object_meta.dx).to(self.device))
    @torch.no_grad()
    def forward(self, object_i, i, norm_constant=None):
        '''Modify object using attenuation correction.

        Parameters
        --------
        object_i : (batch_size, Lx, Ly, Lz) torch.tensor
          This object is such that summation along the x-axis yields the ith projection in
          the image
        i: number
          The projection index
        norm_constant: (batch_size, Lx, Ly, Lz) torch.tensor. Default: None
          If true, modify the norm_constant argument by the normalization factor
          used to scale object_i. This is useful during back projection, where division by
          a normalization constant is required.

        Returns:
          Pytorch tensor of size [batch_size, Lx, Ly, Lz] cor
        '''
        if self.store_in_memory:
            norm_factor = self.probability_matrices[i]
        else:
            norm_factor = get_prob_of_detection_matrix(rotate_detector_z(self.CT, self.image_meta.angles[i]), self.object_meta.dx).to(self.device)
        if norm_constant is not None:
            norm_constant*=norm_factor
        return object_i*norm_factor
# INPUT:
# object [batch_size, Lx, Ly, Lz]
# sigma [Lx] (in pixels) should be same for all objects in batch
# OUTPUT:
# PSF corrected object [batch_size, Lx, Ly, Lz]

class PSFCorrectionNet(nn.Module):
    def __init__(self, object_meta, image_meta, collimator_slope,
                 collimator_intercept, kernel_size=21, kernel_dimensions='2D', device='cpu'):
        super(PSFCorrectionNet, self).__init__()
        self.device = device
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.layers = {}
        for radius in np.unique(image_meta.radii):
            sigma = self.get_sigma(radius, object_meta.dx, object_meta.shape, collimator_slope, collimator_intercept)
            self.layers[radius] = get_PSF_transform(sigma/object_meta.dx, kernel_size, kernel_dimensions=kernel_dimensions, device=self.device)
    def get_sigma(self, radius, dx, shape, collimator_slope, collimator_intercept):
        distances = get_distance(shape[0], radius, dx)
        sigma = collimator_slope * distances + collimator_intercept
        return sigma
    @torch.no_grad()
    def forward(self, object_i, i, norm_constant=None):
        return self.layers[self.image_meta.radii[i]](object_i)

# INPUT:
# object [batch_size, Lx, Ly, Lz]
# angles [batch_size, num_projections]
# radii? [ num_projections] b/c all in batch should have same radii
# CT? [batch_size]
# OUTPUT:
# image [batch_size, num_projections, Ly, Lz]
class ForwardProjectionNet(nn.Module):
    def __init__(self, object_correction_nets, image_correction_nets,
                object_meta, image_meta, device='cpu'):
        super(ForwardProjectionNet, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets
        self.image_correction_nets = image_correction_nets
        self.object_meta = object_meta
        self.image_meta = image_meta

    def forward(self, object, angle_subset=None):
        N_angles = self.image_meta.num_projections
        image = torch.zeros((object.shape[0], N_angles, object.shape[2], object.shape[3])).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = rotate_detector_z(object, self.image_meta.angles[i])
            for net in self.object_correction_nets:
                object_i = net(object_i, i)
            image[:,i] = object_i.sum(axis=1)
        return image
        
# INPUT:
# image [batch_size, num_projections, Ly, Lz]
# Lx 
# angles [batch_size, num_projections]
# radii? [batch_size, num_projections]
# CT? [batch_size]
# OUTPUT:
# object [batch_size, Lx, Ly, Lz]
class BackProjectionNet(nn.Module):
    def __init__(self, object_correction_nets, image_correction_nets,
                 object_meta, image_meta, device='cpu'):
        super(BackProjectionNet, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets.copy()
        self.image_correction_nets = image_correction_nets.copy()
        self.object_correction_nets.reverse()
        self.image_correction_nets.reverse()
        self.object_meta = object_meta
        self.image_meta = image_meta

    def forward(self, image, angle_subset=None, prior=None, delta=1e-11):
        N_angles = self.image_meta.num_projections
        object = torch.zeros([image.shape[0], *self.object_meta.shape]).to(self.device)
        norm_constant = torch.zeros([image.shape[0], *self.object_meta.shape]).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = image[:,i].unsqueeze(dim=1)*torch.ones([image.shape[0], *self.object_meta.shape]).to(self.device)
            norm_constant_i = torch.ones([image.shape[0], *self.object_meta.shape]).to(self.device)
            for net in self.object_correction_nets:
                object_i = net(object_i, i, norm_constant=norm_constant_i)
            norm_constant += rotate_detector_z(norm_constant_i, self.image_meta.angles[i], negative=True)
            object += rotate_detector_z(object_i, self.image_meta.angles[i], negative=True)
        if prior:
            norm_constant += prior()
        return object/(norm_constant + delta)

class OSEMNet(nn.Module):
    def __init__(self, 
                 object_initial,
                 forward_projection_net,
                 back_projection_net,
                 prior = None):
        super(OSEMNet, self).__init__()
        self.forward_projection_net = forward_projection_net
        self.back_projection_net = back_projection_net
        self.object_prediction = object_initial
        self.prior = prior

    def get_subset_splits(self, n_subsets, n_angles):
        indices = np.arange(n_angles).astype(int)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array

    def set_image(self, image):
        self.image = image

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, n_iters, n_subsets, comparisons=None, delta=1e-11):
        subset_indices_array = self.get_subset_splits(n_subsets, self.image.shape[1])
        for j in range(n_iters):
            for subset_indices in subset_indices_array:
                # Set OSL Prior to have object from previous prediction
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                ratio = self.image / (self.forward_projection_net(self.object_prediction, angle_subset=subset_indices) + delta)
                self.object_prediction = self.object_prediction * self.back_projection_net(ratio, angle_subset=subset_indices, prior=self.prior)
                if comparisons:
                    for key in comparisons.keys():
                        comparisons[key].compare(self.object_prediction)
        return self.object_prediction



class SmoothnessPriorOSL(nn.Module):
    def __init__(self, beta, phi, delta=1, device='cpu'):
        super(SmoothnessPriorOSL, self).__init__()
        self.beta = beta
        self.delta = delta
        self.device = device
        self.phi = phi
        self.kernel, self.weights = self.get_kernel()
    def get_kernel(self):
        kernels = []
        weights = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i==1)*(j==1)*(k==1):
                        continue
                    kernel = torch.zeros((3,3,3))
                    kernel[1,1,1] = 1
                    kernel[i,j,k] = -1
                    kernels.append(kernel)
                    weight = 1/np.sqrt((i-1)**2 + (j-1)**2 + (k-1)**2)
                    weights.append(weight)
        kern = torch.nn.Conv3d(1, 26, 3, padding='same', padding_mode='reflect', bias=0, device=self.device)
        kern.weight.data = torch.stack(kernels).unsqueeze(dim=1).to(self.device)
        weights = torch.tensor(weights).to(self.device)
        return kern, weights
    def set_object(self, object):
        self.object = object
    @torch.no_grad()
    def forward(self):
        phis = self.phi(self.kernel(self.object.unsqueeze(dim=1))/self.delta)
        all_summation_terms = phis * self.weights.view(-1,1,1,1)
        return self.beta/self.delta * all_summation_terms.sum(axis=1)

class QuadraticPriorOSL(SmoothnessPriorOSL):
    def __init__(self, beta, device='cpu'):
        super(QuadraticPriorOSL, self).__init__(beta, lambda x: x, device=device)







# RENAME ALL THIS LATER
class CompareToNumber():
    def __init__(self, number, mask, norm_factor=None):
        self.number = number
        self.mask = mask
        self.biass = []
        self.vars = []
        self.norm_factor = norm_factor
    def compare(self, prediction):
        prediction = torch.clone(prediction)
        if self.norm_factor:
            prediction *= self.norm_factor 
        self.biass.append(torch.mean(prediction[self.mask] - self.number).item())
        self.vars.append(torch.var(prediction[self.mask] - self.number).item())
