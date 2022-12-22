import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np


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
    return d

def get_PSF_transform(sigma, kernel_size, delta=1e-9, device='cpu'):
    N = len(sigma)
    layer = torch.nn.Conv2d(N, N, kernel_size, groups=N, padding='same',
                            padding_mode='replicate', bias=0, device=device)
    x_grid, y_grid = torch.meshgrid(2*[torch.arange(-int(kernel_size//2), int(kernel_size//2)+1)],
                                    indexing='ij')
    x_grid = x_grid.unsqueeze(dim=0).repeat((N,1,1))
    y_grid = y_grid.unsqueeze(dim=0).repeat((N,1,1))
    sigma = torch.tensor(sigma, dtype=torch.float32).reshape((N,1,1))
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2*sigma**2 + delta))
    kernel = kernel / kernel.sum(axis=(1,2)).reshape(N,1,1)
    layer.weight.data = kernel.unsqueeze(dim=1).to(device)
    return layer

# INPUT: 
# object [batch_size, Lx, Ly, Lz] 
# CT [batch_size, Lx, Ly, Lz]
# angle
# OUTPUT: 
# CT corrected object [batch_size, Lx, Ly, Lz]

class CTCorrectionNet(nn.Module):
    def __init__(self, CT, angle, dx, device='cpu'):
        super(CTCorrectionNet, self).__init__()
        self.CT = CT
        self.angle = angle
        self.dx = dx
        self.device = device
    @torch.no_grad()
    def forward(self, object, return_norm_factor=False):
        norm_factor = get_prob_of_detection_matrix(rotate_detector_z(self.CT, self.angle), self.dx).to(self.device)
        if return_norm_factor:
            return object*norm_factor, norm_factor
        else:
            return object*norm_factor
# INPUT:
# object [batch_size, Lx, Ly, Lz]
# sigma [Lx] (in pixels) should be same for all objects in batch
# OUTPUT:
# PSF corrected object [batch_size, Lx, Ly, Lz]

class PSFCorrectionNet(nn.Module):
    def __init__(self, radius, dx, shape, collimator_slope,
                 collimator_intercept, kernel_size=41, device='cpu'):
        super(PSFCorrectionNet, self).__init__()
        self.device = device
        sigma = self.get_sigma(radius, dx, shape, collimator_slope, collimator_intercept)
        self.layer = get_PSF_transform(sigma/dx, kernel_size, device=self.device)

    def get_sigma(self, radius, dx, shape, collimator_slope, collimator_intercept):
        distances = get_distance(shape[0], radius, dx)
        sigma = collimator_slope * distances + collimator_intercept
        return sigma
    @torch.no_grad()
    def forward(self, object):
        return self.layer(object)

# INPUT:
# object [batch_size, Lx, Ly, Lz]
# angles [batch_size, num_projections]
# radii? [ num_projections] b/c all in batch should have same radii
# CT? [batch_size]
# OUTPUT:
# image [batch_size, num_projections, Ly, Lz]
class ForwardProjectionNet(nn.Module):
    def __init__(self, angles, dx, shape, radii=None, CT=None,
                 collimator_slope=None, collimator_intercept=None, PSF_net_kernel_size = 21, device='cpu',
                 ):
        super(ForwardProjectionNet, self).__init__()
        self.CT_correction = CT is not None
        self.PSF_correction = radii is not None
        self.angles = angles
        self.shape = shape
        self.device = device
        if self.CT_correction:
            self.ct_correction_nets = [CTCorrectionNet(CT, angle, dx, device=self.device) for angle in angles]
        if self.PSF_correction:
            self.psf_correction_nets = [PSFCorrectionNet(radius, dx, self.shape, collimator_slope, collimator_intercept, kernel_size = PSF_net_kernel_size, device=self.device) for radius in radii]

    def forward(self, object):
        image = torch.zeros((object.shape[0], len(self.angles), object.shape[2], object.shape[3])).to(self.device)
        for i, angle in enumerate(self.angles):
            object_i = rotate_detector_z(object, angle)
            if self.CT_correction:
                object_i = self.ct_correction_nets[i](object_i)
            if self.PSF_correction:
                object_i = self.psf_correction_nets[i](object_i)
            
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
    def __init__(self, angles, dx, shape, radii=None, CT=None,
                 collimator_slope=None, collimator_intercept=None, PSF_net_kernel_size = 21, device='cpu'):
        super(BackProjectionNet, self).__init__()
        self.CT_correction = CT is not None
        self.PSF_correction = radii is not None
        self.angles = angles
        self.shape = shape
        self.device = device
        if self.CT_correction:
            self.ct_correction_nets = [CTCorrectionNet(CT, angle, dx, device=self.device) for angle in angles]
        if self.PSF_correction:
            self.psf_correction_nets = [PSFCorrectionNet(radius, dx, self.shape, collimator_slope, collimator_intercept, kernel_size = PSF_net_kernel_size, device=self.device) for radius in radii]

    def forward(self, image, delta=1e-9):
        object = torch.zeros([image.shape[0], *self.shape]).to(self.device)
        norm_constant = torch.zeros([image.shape[0], *self.shape]).to(self.device)
        for i, angle in enumerate(self.angles):
            object_i = image[:,i]*torch.ones([image.shape[0], *self.shape]).to(self.device)
            if self.PSF_correction:
                object_i = self.psf_correction_nets[i](object_i)
                #norm_constant_i = self.psf_correction_nets[i](norm_constant_i)
            if self.CT_correction:
                object_i, norm_constant_i = self.ct_correction_nets[i](object_i, return_norm_factor=True)
            else:
                norm_constant_i = torch.ones([image.shape[0], *self.shape]).to(self.device)
            norm_constant += rotate_detector_z(norm_constant_i, angle, negative=True)
            object += rotate_detector_z(object_i, angle, negative=True)
        return object/(norm_constant + delta)

class OSEMNet(nn.Module):
    def __init__(self, object_initial, angles, dx, shape, radii=None, CT=None,
                 collimator_slope=None, collimator_intercept=None, PSF_net_kernel_size=21, device='cpu',
                 ):
        super(OSEMNet, self).__init__()
        self.forward_projection_net = ForwardProjectionNet(angles, dx, shape, radii, CT,
                 collimator_slope, collimator_intercept, PSF_net_kernel_size, device)
        self.back_projection_net = BackProjectionNet(angles, dx, shape, radii, CT,
                 collimator_slope, collimator_intercept, PSF_net_kernel_size, device)
        self.object_prediction = object_initial
    def forward(self, image, n_iters, delta=1e-9):
        for i in range(n_iters):
            ratio = image / (self.forward_projection_net(self.object_prediction) + delta)
            self.object_prediction = self.object_prediction * self.back_projection_net(ratio)
        return self.object_prediction