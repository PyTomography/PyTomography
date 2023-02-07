import torch
import torch.nn as nn
import numpy as np

class SmoothnessPrior(nn.Module):
    r"""Implementation of priors with gradients of the form 
    :math:`\frac{\partial V}{\partial f_r}=\frac{\beta}{\delta}\sum_{r,s}w_{s}\phi\left(\frac{f_r-f_s}{\delta}\right)` where :math:`V` is from the log-posterior probability :math:`\log P(g | f) - \beta V(f)`.
    """
    def __init__(self, beta, phi, delta=1, device='cpu'):
        """Initializer

        Args:
            beta (float): Used to scale the weight of the prior
            phi (function): Function $\phi$ used in formula above
            delta (int, optional): Parameter $\delta$ in equation above. Defaults to 1.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
        """ 
        super(SmoothnessPrior, self).__init__()
        self.beta = beta
        self.beta_scale_factor = 1
        self.delta = delta
        self.device = device
        self.phi = phi

    def get_kernel(self):
        r"""Obtains the kernel used to get :math:`\frac{\partial V}{\partial f_r}` (this is an array with the same dimensions as the object space image)

        Returns:
            (torch.nn.Conv3d, torch.tensor): Kernel used for convolution (number of output channels equal to number of :math:`s`), and array of weights :math:`w_s` used in expression for gradient.
        """
        dx, dy, dz = self.object_meta.dr
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
                    weight = dx/np.sqrt((dx*(i-1))**2 + (dy*(j-1))**2 + (dz*(k-1))**2)
                    weights.append(weight)
        kern = torch.nn.Conv3d(1, 26, 3, padding='same', padding_mode='reflect', bias=0, device=self.device)
        kern.weight.data = torch.stack(kernels).unsqueeze(dim=1).to(self.device)
        weights = torch.tensor(weights).to(self.device)
        return kern, weights

    def set_object_meta(self, object_meta):
        """Sets object metadata parameters.

        Args:
            object_meta (ObjectMeta): Object metadata describing the system.
        """
        self.object_meta = object_meta

    def set_kernel(self, object_meta):
        """Sets the kernel using  `get_kernel` and the corresponding object metadata.

        Args:
            object_meta (_type_): _description_
        """
        self.set_object_meta(object_meta)
        self.kernel, self.weights = self.get_kernel()

    def set_beta_scale(self, factor):
        r"""Sets :math:`\beta` 

        Args:
            factor (float): Value of :math:`\beta` 
        """
        self.beta_scale_factor = factor

    def set_object(self, object):
        """Sets the object :math:`f_r` used to compute :math:`\frac{\partial V}{\partial f_r}` 

        Args:
            object (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] which the prior
            will be computed on
        """
        self.object = object

    def set_device(self, device):
        """Sets the pytorch computation device

        Args:
            device (str): sets device.
        """
        self.device=device

    @torch.no_grad()
    def forward(self):
        r"""Computes the prior on self.object

        Returns:
            torch.tensor: Tensor of shape [batch_size, Lx, Ly, Lz] representing :math:`\frac{\partial V}{\partial f_r}`
        """
        phis = self.phi(self.kernel(self.object.unsqueeze(dim=1))/self.delta)
        all_summation_terms = phis * self.weights.view(-1,1,1,1)
        return self.beta*self.beta_scale_factor/self.delta * all_summation_terms.sum(axis=1)

class QuadraticPrior(SmoothnessPrior):
    r"""Implentation of `SmoothnessPrior` where :math:`\phi` is the identity function"""
    def __init__(self, beta, delta=1, device='cpu'):
        super(QuadraticPrior, self).__init__(beta, lambda x: x, delta=delta, device=device)

class LogCoshPrior(SmoothnessPrior):
    r"""Implementation of `SmoothnessPrior` where :math:`\phi` is the hyperbolic tangent function"""
    def __init__(self, beta, delta=1, device='cpu'):
        super(LogCoshPrior, self).__init__(beta, torch.tanh, delta=delta, device=device)