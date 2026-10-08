from __future__ import annotations
from typing import Callable
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


class FBPFilter:
    r"""Window of filtered back projection. The ramp :math:`|f|` is part of the reconstruction; the filter multiplies it
    by a window :math:`W(f)`. A window is called with the spatial frequency :math:`f` (cycles per mm, a tensor) and the
    Nyquist frequency :math:`f_N` of the sampling it is applied to (cycles per mm), so that each system matrix can apply
    it in its own geometry."""
    def __call__(self, f: torch.Tensor, f_nyquist: float) -> torch.Tensor:
        raise NotImplementedError


class RamLakFilter(FBPFilter):
    r"""No window, :math:`W(f) = 1`: the plain ramp (Ram-Lak). The sharpest and the noisiest."""
    def __call__(self, f, f_nyquist):
        return torch.ones_like(f)


class SheppLoganFilter(FBPFilter):
    r"""Shepp-Logan window :math:`W(f) = \mathrm{sinc}(f / 2f_N)`, which is :math:`2/\pi` at the Nyquist frequency."""
    def __call__(self, f, f_nyquist):
        return torch.sinc(f / (2 * f_nyquist))


class GeneralizedHammingFilter(FBPFilter):
    r"""Window :math:`W(f) = \alpha + (1 - \alpha)\cos(\pi f / f_c)` for :math:`|f| \le f_c` and 0 above it, with the
    cut-off :math:`f_c` a fraction ``cutoff`` of the Nyquist frequency. :math:`\alpha = 0.5` is the Hann window and
    :math:`\alpha = 0.54` the Hamming window.

    Args:
        alpha (float): Window parameter.
        cutoff (float, optional): Cut-off as a fraction of the Nyquist frequency. Defaults to 1.
    """
    def __init__(self, alpha: float, cutoff: float = 1.0):
        self.alpha, self.cutoff = alpha, cutoff

    def __call__(self, f, f_nyquist):
        fc = self.cutoff * f_nyquist
        w = self.alpha + (1 - self.alpha) * torch.cos(np.pi * f / fc)
        return torch.where(f.abs() <= fc, w, torch.zeros_like(w))


class HannFilter(GeneralizedHammingFilter):
    r"""Hann window :math:`W(f) = \frac{1}{2}(1 + \cos(\pi f / f_c))`, zero from the cut-off on (by default the Nyquist
    frequency).

    Args:
        cutoff (float, optional): Cut-off as a fraction of the Nyquist frequency. Defaults to 1.
    """
    def __init__(self, cutoff: float = 1.0):
        super().__init__(0.5, cutoff)


class CosineFilter(FBPFilter):
    r"""Cosine window :math:`W(f) = \cos(\pi f / 2 f_c)`, zero from the cut-off on (by default the Nyquist frequency).

    Args:
        cutoff (float, optional): Cut-off as a fraction of the Nyquist frequency. Defaults to 1.
    """
    def __init__(self, cutoff: float = 1.0):
        self.cutoff = cutoff

    def __call__(self, f, f_nyquist):
        fc = self.cutoff * f_nyquist
        w = torch.cos(np.pi * f / (2 * fc))
        return torch.where(f.abs() <= fc, w, torch.zeros_like(w))


class TabulatedFilter(FBPFilter):
    r"""A window given as a table against spatial frequency, for example a scanner's kernel measured relative to the
    plain ramp. It is interpolated linearly; beyond the last tabulated frequency, the last value falls linearly to zero
    over ``taper`` cycles per mm.

    Args:
        frequencies (array): Spatial frequencies (cycles per mm), increasing.
        values (array): Window at those frequencies.
        taper (float, optional): Width (cycles per mm) of the fall to zero after the table. Defaults to 0.1.
    """
    def __init__(self, frequencies, values, taper: float = 0.1):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.taper = taper

    def __call__(self, f, f_nyquist):
        a = np.abs(f.detach().cpu().numpy().astype(np.float64))
        w = np.interp(a, self.frequencies, self.values)
        last = self.frequencies[-1]
        w = np.where(a <= last, w, self.values[-1] * np.clip(1 - (a - last) / self.taper, 0, 1))
        return torch.as_tensor(w, dtype=f.dtype, device=f.device)


class _FunctionFilter(FBPFilter):
    """A user function of the spatial frequency alone (cycles per mm)."""
    def __init__(self, function: Callable):
        self.function = function

    def __call__(self, f, f_nyquist):
        return torch.as_tensor(self.function(f), dtype=f.dtype, device=f.device)


class _CyclesPerSampleFilter(FBPFilter):
    """The older filters, :class:`RampFilter` and :class:`HammingFilter`, which are called with the frequency in cycles
    per sample."""
    def __init__(self, filter):
        self.filter = filter

    def __call__(self, f, f_nyquist):
        if isinstance(self.filter, RampFilter):
            return torch.ones_like(f)
        return torch.as_tensor(self.filter(f / (2 * f_nyquist)), dtype=f.dtype, device=f.device)


_NAMED_FILTERS = {'ram-lak': RamLakFilter, 'ramlak': RamLakFilter, 'ramp': RamLakFilter, 'shepp-logan': SheppLoganFilter,
                  'hann': HannFilter, 'hanning': HannFilter, 'hamming': lambda: GeneralizedHammingFilter(0.54),
                  'cosine': CosineFilter}


def get_fbp_filter(filter) -> FBPFilter:
    """The window of filtered back projection described by ``filter``: a name (``'ram-lak'``, ``'shepp-logan'``,
    ``'hann'``, ``'hamming'`` or ``'cosine'``), an :class:`FBPFilter`, a function of the spatial frequency in cycles per
    mm, or one of the older :class:`RampFilter` and :class:`HammingFilter` (class or instance). None means Ram-Lak.

    Args:
        filter: Description of the filter.

    Returns:
        FBPFilter: A window, called as ``window(f, f_nyquist)``.
    """
    if filter is None:
        return RamLakFilter()
    if isinstance(filter, str):
        try:
            return _NAMED_FILTERS[filter.lower()]()
        except KeyError:
            raise ValueError(f'unknown filter {filter!r}: use one of {sorted(_NAMED_FILTERS)}, an FBPFilter, or a function of frequency') from None
    if isinstance(filter, FBPFilter):
        return filter
    if filter is RampFilter:
        return RamLakFilter()
    if filter is HammingFilter:
        return HannFilter()
    if isinstance(filter, (RampFilter, HammingFilter)):
        return _CyclesPerSampleFilter(filter)
    if callable(filter):
        return _FunctionFilter(filter)
    raise TypeError(f'cannot use {filter!r} as a filter')
