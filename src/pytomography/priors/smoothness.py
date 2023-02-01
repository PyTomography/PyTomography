import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, beta, delta=1, device='cpu'):
        super(QuadraticPriorOSL, self).__init__(beta, lambda x: x, delta=delta, device=device)

class LogCoshPriorOSL(SmoothnessPriorOSL):
    def __init__(self, beta, delta=1, device='cpu'):
        super(LogCoshPriorOSL, self).__init__(beta, torch.tanh, delta=delta, device=device)