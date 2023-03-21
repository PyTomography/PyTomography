import torch
import torch.nn as nn
import numpy as np
from .prior import Prior
from pytomography.metadata import ObjectMeta
from collections.abc import Callable

class SmoothnessPrior(Prior):
    r"""Implementation of priors with gradients of the form :math:`\frac{\partial V}{\partial f_r}=\frac{\beta}{\delta}\sum_{s}w_{r,s}\phi\left(\frac{f_r-f_s}{\delta}\right)` where :math:`V` is from the log-posterior probability :math:`\ln L (\tilde{f}, f) - \beta V(f)`.
    
    Args:
            beta (float): Used to scale the weight of the prior
            phi (function): Function :math:`\phi` used in formula above
            delta (int, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
    """
    def __init__(
        self,
        beta: float,
        delta: float,
        phi: Callable,
        device: str = 'cpu'
    ) -> None:
        super(SmoothnessPrior, self).__init__(beta, device)
        self.delta = delta
        self.phi = phi

    def get_kernel(self) -> torch.nn.Conv3d:
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

    def set_kernel(self, object_meta: ObjectMeta) -> None:
        """Sets the kernel using  `get_kernel` and the corresponding object metadata.

        Args:
            object_meta (_type_): _description_
        """
        self.set_object_meta(object_meta)
        self.kernel, self.weights = self.get_kernel()

    @torch.no_grad()
    def forward(self) -> torch.tensor:
        r"""Computes the prior on self.object

        Returns:
            torch.tensor: Tensor of shape [batch_size, Lx, Ly, Lz] representing :math:`\frac{\partial V}{\partial f_r}`
        """
        phis = self.phi(self.kernel(self.object.unsqueeze(dim=1))/self.delta)
        all_summation_terms = phis * self.weights.view(-1,1,1,1)
        return self.beta*self.beta_scale_factor/self.delta * all_summation_terms.sum(axis=1)

class QuadraticPrior(SmoothnessPrior):
    r"""Subclass of `SmoothnessPrior` where :math:`\phi(x)=x` corresponds to a quadratic prior :math:`V(f)=\frac{1}{4}\sum_{r,s} w_{r,s} \left(\frac{f_r-f_s}{\delta}\right)^2`
    
    Args:
            beta (float): Used to scale the weight of the prior
            delta (int, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
        device: str = 'cpu'
    ) -> None:
        super(QuadraticPrior, self).__init__(beta, delta, lambda x: x, device=device)

class LogCoshPrior(SmoothnessPrior):
    r"""Subclass of `SmoothnessPrior` where :math:`\phi(x)=\tanh(x)` corresponds to the logcosh prior :math:`V(f)=\sum_{r,s} w_{r,s} \log\cosh\left(\frac{f_r-f_s}{\delta}\right)`
    
    Args:
            beta (float): Used to scale the weight of the prior
            delta (int, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
        device: str = 'cpu'
    ) -> None:
        super(LogCoshPrior, self).__init__(beta, delta, torch.tanh, device=device)


