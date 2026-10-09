from __future__ import annotations
from typing import Callable
import torch
import numpy as np


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


_NAMED_FILTERS = {'ram-lak': RamLakFilter, 'ramlak': RamLakFilter, 'ramp': RamLakFilter, 'shepp-logan': SheppLoganFilter,
                  'hann': HannFilter, 'hanning': HannFilter, 'hamming': lambda: GeneralizedHammingFilter(0.54),
                  'cosine': CosineFilter}


def get_fbp_filter(filter) -> FBPFilter:
    """The window of filtered back projection described by ``filter``: a name (``'ram-lak'``, ``'shepp-logan'``,
    ``'hann'``, ``'hamming'`` or ``'cosine'``), an :class:`FBPFilter` (instance or class), or a function of the spatial
    frequency in cycles per mm. None means Ram-Lak.

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
    if isinstance(filter, type) and issubclass(filter, FBPFilter):
        return filter()
    if callable(filter):
        return _FunctionFilter(filter)
    raise TypeError(f'cannot use {filter!r} as a filter')


def ramp_filter_response(n: int, spacing: float, window: FBPFilter | None = None) -> torch.Tensor:
    r"""Frequency response, in FFT order, of the band-limited ramp (Ram-Lak) filter for samples ``spacing`` apart, times
    a window: the DFT of the kernel :math:`h_0 = 1/(4s^2)`, :math:`h_k = -1/(\pi k s)^2` for odd :math:`k` and 0 for even
    :math:`k \neq 0` (Kak and Slaney, ch. 3), which follows :math:`|f|` but, unlike :math:`|f|` sampled in
    frequency, has no error at low frequencies. Filter with ``spacing * ifft(fft(p, n) * response)``, with ``n`` at least twice the
    number of samples, so that nothing wraps around.

    Args:
        n (int): Length of the (zero padded) FFT.
        spacing (float): Sample spacing. The window is given the frequency in cycles per unit of it, so the windows of
            :func:`get_fbp_filter` need it in mm.
        window (FBPFilter, optional): Window, called as ``window(f, f_nyquist)``. Defaults to None (none).

    Returns:
        torch.Tensor: The response (``n`` values, float64, on the host).
    """
    spacing = float(spacing)
    k = torch.arange(-(n // 2), n - n // 2, dtype=torch.float64)
    h = torch.zeros_like(k)
    h[k == 0] = 1 / (4 * spacing ** 2)
    odd = (k.abs() % 2) == 1
    h[odd] = -1 / (np.pi ** 2 * (k[odd] * spacing) ** 2)
    response = torch.fft.fft(torch.fft.ifftshift(h)).real
    if window is not None:
        f = torch.fft.fftfreq(n, d=spacing).abs().to(torch.float64)
        response = response * window(f, 0.5 / spacing).to(torch.float64).cpu()
    return response


def ramp_filter(projections: torch.Tensor, spacing: float, window: FBPFilter | None = None, dim: int = -1) -> torch.Tensor:
    r"""Ramp filters ``projections`` along ``dim``: linear convolution with the kernel of :func:`ramp_filter_response`
    (zero padded to a power of two at least twice as long), times the window. The result is in the units of the
    projections per unit of ``spacing``.

    Args:
        projections (torch.Tensor): Projections, sampled ``spacing`` apart along ``dim``.
        spacing (float): Sample spacing (mm for the windows of :func:`get_fbp_filter`).
        window (FBPFilter, optional): Window, called as ``window(f, f_nyquist)``. Defaults to None (none).
        dim (int, optional): Dimension to filter along. Defaults to -1.

    Returns:
        torch.Tensor: The filtered projections, the shape of ``projections``.
    """
    n = projections.shape[dim]
    n_pad = int(2 ** np.ceil(np.log2(2 * n)))
    response = ramp_filter_response(n_pad, spacing, window)[:n_pad // 2 + 1].to(projections.device, torch.float32)
    shape = [1] * projections.ndim
    shape[dim] = n_pad // 2 + 1
    spectrum = torch.fft.rfft(projections.float(), n=n_pad, dim=dim) * response.reshape(shape)
    return float(spacing) * torch.fft.irfft(spectrum, n=n_pad, dim=dim).narrow(dim, 0, n)
