"""Fused CUDA kernel for the time of flight weighting of the single scatter simulation (``sss._tof_weighted_emission``),
compiled at run time with CuPy (NVRTC), used when CuPy is installed and the device is a CUDA GPU.

For each scatter point, the PyTorch version builds the Gaussian TOF kernel between every sampled LOR, TOF bin and piece
of the LOR as one [LORs, TOF bins, pieces] tensor (850 MB for the GATE mMR with 21 TOF bins, 25 pieces and every sixth
ring and crystal), normalises it over the bins and sums it over the pieces: a few passes over that tensor per scatter
point, so it is bound by memory traffic (31 s for the tutorial's 2122 scatter points on an RTX 5090). Here one thread
holds one LOR: its kernel values stay in registers one piece at a time, and the piece centres and emission are read from
per-crystal tables. The kernel values are computed exactly as PyTorch computes them; only the sums over the TOF bins and
over the pieces run in a different order, which changes the result in the last bits of float32.
"""
from __future__ import annotations
import numpy as np
import torch

#: Most TOF bins the kernel holds in registers.
MAX_TOF_BINS = 64

_SOURCE = r'''
extern "C" __global__ void tof_weighted_emission(
    const float* __restrict__ offset, const long long* __restrict__ crystal, const float* __restrict__ centers,
    const float* __restrict__ emission, float* __restrict__ out, int L, int P, float coef)
{
    // offset [T_BINS, L]: TOF offset of each LOR in each bin; crystal [L]: row of each LOR in the per-crystal tables;
    // centers, emission [crystals, P]: distance along the line of each piece's centre, and its emission integral;
    // out [T_BINS, L]; coef = -0.5 / sigma^2
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= L) return;
    float o[T_BINS], acc[T_BINS], k[T_BINS];
    #pragma unroll
    for (int t = 0; t < T_BINS; ++t) { o[t] = offset[(long long)t * L + l]; acc[t] = 0.f; }
    const float* c = centers + crystal[l] * P;
    const float* e = emission + crystal[l] * P;
    for (int p = 0; p < P; ++p) {
        float cp = c[p], norm = 0.f;
        #pragma unroll
        for (int t = 0; t < T_BINS; ++t) { float d = o[t] - cp; k[t] = expf((d * d) * coef); norm += k[t]; }
        // a piece far from every TOF bin has a kernel of zero in all of them: it contributes nothing (rather than 0/0)
        float w = norm > 0.f ? e[p] / norm : 0.f;
        #pragma unroll
        for (int t = 0; t < T_BINS; ++t) acc[t] += k[t] * w;
    }
    #pragma unroll
    for (int t = 0; t < T_BINS; ++t) out[(long long)t * L + l] = acc[t];
}
'''

_KERNELS = {}


def available(device, num_tof_bins: int) -> bool:
    """Whether the fused kernel can run on ``device`` for ``num_tof_bins`` TOF bins: a CUDA device, with CuPy installed."""
    if torch.device(device).type != 'cuda' or num_tof_bins > MAX_TOF_BINS:
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _kernel(num_tof_bins: int):
    """The kernel compiled for ``num_tof_bins`` TOF bins (its loops over the bins are unrolled, so its arrays stay in registers)."""
    if num_tof_bins not in _KERNELS:
        import cupy
        _KERNELS[num_tof_bins] = cupy.RawKernel(_SOURCE, 'tof_weighted_emission', options=(f'-DT_BINS={num_tof_bins}',))
    return _KERNELS[num_tof_bins]


class _StreamHandle:
    """A raw CUDA stream, through the CUDA stream protocol (PyTorch's streams do not implement it yet)."""
    def __init__(self, handle: int):
        self.handle = handle

    def __cuda_stream__(self):
        return (0, self.handle)


def _torch_stream(device):
    """The current PyTorch stream on ``device``, as a CuPy stream, so that the kernel queues behind PyTorch's work."""
    import cupy
    handle = torch.cuda.current_stream(device).cuda_stream
    if hasattr(cupy.cuda.Stream, 'from_external'):       # CuPy 14 deprecates ExternalStream for it
        return cupy.cuda.Stream.from_external(_StreamHandle(handle))
    return cupy.cuda.ExternalStream(handle)


def tof_weighted_emission(offset: torch.Tensor, crystal: torch.Tensor, centers: torch.Tensor, emission: torch.Tensor, sigma: float) -> torch.Tensor:
    """``sss._tof_weighted_emission(offset, centers[crystal], emission[crystal], sigma, ...)`` in one fused kernel.

    Args:
        offset (torch.Tensor): [TOF bins, LORs] TOF offsets of the LORs.
        crystal (torch.Tensor): [LORs] row of each LOR in ``centers`` and ``emission``.
        centers (torch.Tensor): [crystals, pieces] distance of each piece's centre along its line.
        emission (torch.Tensor): [crystals, pieces] emission integrals of the pieces.
        sigma (float): TOF resolution, as a standard deviation in spatial units.

    Returns:
        torch.Tensor: [TOF bins, LORs] emission seen by each TOF bin.
    """
    import cupy
    device = offset.device
    T, L = offset.shape
    offset = offset.to(torch.float32).contiguous()
    crystal = crystal.to(device=device, dtype=torch.int64).contiguous()
    centers = centers.to(device=device, dtype=torch.float32).contiguous()
    emission = emission.to(device=device, dtype=torch.float32).contiguous()
    out = torch.empty((T, L), dtype=torch.float32, device=device)
    if L == 0:
        return out
    threads = 128
    with _torch_stream(device):
        _kernel(T)(((L + threads - 1) // threads,), (threads,), (
            cupy.asarray(offset), cupy.asarray(crystal), cupy.asarray(centers), cupy.asarray(emission), cupy.asarray(out),
            np.int32(L), np.int32(centers.shape[1]), np.float32(-0.5 / sigma**2)))
    return out
