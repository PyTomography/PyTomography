import torch
import torch.nn as nn
import numpy as np
from .prior import Prior
from pytomography.metadata import ObjectMeta
from collections.abc import Callable

class DiffAndSumSmoothnessPrior(Prior):
    r"""Implementation of priors where gradients depend on difference and the sum of neighbouring voxels:
    :math:`\frac{\partial V}{\partial f_r}=\frac{\beta}{\delta}\sum_{r,s}w_{s}\phi\left(\frac{f_r-f_s}{\delta}\right)` where :math:`V` is from the log-posterior probability :math:`\log P(g | f) - \beta V(f)`.
    """
    def __init__(
        self,
        beta: float,
        phi: Callable, 
        device: str = 'cpu', 
        **kwargs
    ) -> None:
        """Initializer

        Args:
            beta (float): Used to scale the weight of the prior
            phi (function): Function $\phi$ used in formula above
            delta (int, optional): Parameter $\delta$ in equation above. Defaults to 1.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
        """ 
        super(DiffAndSumSmoothnessPrior, self).__init__(beta, device)
        self.phi = phi
        self.kwargs = kwargs

    def get_kernel(
        self,
        sign: float = 1
    ) -> torch.nn.Conv3d:
        r"""Obtains the kernel used to get :math:`\frac{\partial V}{\partial f_r}` (this is an array with the same dimensions as the object space image)

        Args:
            sign (float): Kernel computes image :math:`f_r + \text{sign} \cdot f_k` for all 26 nearest neighbours :math:`k` (i.e. a 3D image is returned with 26 channels). Defaults to 1.

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
                    kernel[i,j,k] = sign
                    kernels.append(kernel)
                    weight = dx/np.sqrt((dx*(i-1))**2 + (dy*(j-1))**2 + (dz*(k-1))**2)
                    weights.append(weight)
        kern = torch.nn.Conv3d(1, 26, 3, padding='same', padding_mode='reflect', bias=0, device=self.device)
        kern.weight.data = torch.stack(kernels).unsqueeze(dim=1).to(self.device)
        weights = torch.tensor(weights).to(self.device)
        return kern.to(torch.float32), weights.to(torch.float32)

    def set_kernel(self, object_meta: ObjectMeta) -> None:
        """Sets the kernel using  `get_kernel` and the corresponding object metadata.

        Args:
            object_meta (ObjectMeta): Metadata for object space. 
        """
        self.set_object_meta(object_meta)
        self.kernel_add, self.weights_add = self.get_kernel(sign=1)
        self.kernel_sub, self.weights_sub = self.get_kernel(sign=-1)

    @torch.no_grad()
    def forward(self) -> torch.tensor:
        r"""Computes the prior on ``self.object``

        Returns:
            torch.tensor: Tensor of shape [batch_size, Lx, Ly, Lz] representing :math:`\frac{\partial V}{\partial f_r}`
        """
        phis = self.phi(self.kernel_add(self.object.unsqueeze(dim=1)), self.kernel_sub(self.object.unsqueeze(dim=1)), **self.kwargs)
        all_summation_terms = phis * self.weights_add.view(-1,1,1,1)
        return self.beta*self.beta_scale_factor * all_summation_terms.sum(axis=1)
    

class QClearPrior(DiffAndSumSmoothnessPrior):
    r"""Implentation of `SmoothnessPrior` where :math:`\phi` is the the QClear Function (DEFINE HERE)"""
    def __init__(
        self, 
        beta: float = 1, 
        gamma: float = 1, 
        device: str ='cpu'
    ) -> None:
        super(QClearPrior, self).__init__(beta, QClear, gamma=gamma, device=device)

def QClear(sum, diff, gamma, eps=1e-11):
        return 4*sum*diff / (sum + gamma*torch.abs(diff) + eps)**2