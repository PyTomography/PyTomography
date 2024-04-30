from __future__ import annotations
import pytomography
import torch
import numpy as np

class RampFilter:
    r"""Implementation of the Ramp filter :math:`\Pi(\omega) = |\omega|`
    """
    def __init__(self):
        return
    def __call__(self, w):
        return torch.abs(w)

class HammingFilter:
    r"""Implementation of the Hamming filter given by :math:`\Pi(\omega) = \frac{1}{2}\left(1+\cos\left(\frac{\pi(|\omega|-\omega_L)}{\omega_H-\omega_L} \right)\right)` for :math:`\omega_L \leq |\omega| < \omega_H` and :math:`\Pi(\omega) = 1` for :math:`|\omega| \leq \omega_L` and :math:`\Pi(\omega) = 0` for :math:`|\omega|>\omega_H`. Arguments ``wl`` and ``wh`` should be expressed as fractions of the Nyquist frequency (i.e. ``wh=0.93`` represents 93% the Nyquist frequency).
    """
    def __init__(self, wl, wh):
        self.wl = wl/2 # units of Nyquist Frequency
        self.wh = wh/2
    def __call__(self, w):
        w = w.cpu().numpy()
        filter = np.piecewise(
        w,
        [np.abs(w)<=self.wl, (self.wl<np.abs(w))*(self.wh>=np.abs(w)), np.abs(w)>self.wh],
        [lambda w: 1, lambda w: 1/2*(1+np.cos(np.pi*(np.abs(w)-self.wl)/(self.wh-self.wl))), lambda w: 0])
        return torch.tensor(filter).to(pytomography.device)
    
