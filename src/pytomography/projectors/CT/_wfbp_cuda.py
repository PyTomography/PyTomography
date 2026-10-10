"""Fused CUDA kernel for the WFBP back projection of :mod:`._wfbp`, compiled at run time with CuPy (NVRTC), used when
CuPy is installed and the device is a CUDA GPU.

The PyTorch back projection builds each step (ray geometry, row coordinates, row weights, their sum over the rays half
a turn apart, sampling) as a separate pass over (views x Nx x Ny x z) temporaries, so it is bound by memory traffic.
Here one thread holds one voxel: for each view of a batch it rejects the view at once if the voxel is outside its cone,
and otherwise computes the rays at theta + k pi in registers, weights them, and samples the filtered projection once.
The voxel is written once per batch. GPU memory is the output, an accumulator of the same size, and one batch of
filtered projections.
"""
from __future__ import annotations
import numpy as np
import torch
from pytomography.utils.memory import gpu_budget
from . import _wfbp

#: Most partner rays either side (k_range) the kernel holds in registers.
MAX_K_RANGE = 8

_SOURCE = r'''
#define PI_F 3.14159265358979f
#define MAXP 17
__device__ __forceinline__ float bilinear(const float* Qj, float ti, float ri, int M, int nrow) {
    // grid_sample(mode='bilinear', align_corners=True, padding_mode='zeros') of Qj (M, nrow) at (ti, ri)
    int t0 = (int)floorf(ti), r0 = (int)floorf(ri);
    float ft = ti - t0, fr = ri - r0, v = 0.f;
    for (int a = 0; a < 2; ++a) {
        int ii = t0 + a;
        if (ii < 0 || ii >= M) continue;
        float wt = a ? ft : 1.f - ft;
        for (int b = 0; b < 2; ++b) {
            int jj = r0 + b;
            if (jj < 0 || jj >= nrow) continue;
            v += wt * (b ? fr : 1.f - fr) * Qj[(long long)ii * nrow + jj];
        }
    }
    return v;
}
extern "C" __global__ void wfbp_bp(
    const float* __restrict__ Q, const float* __restrict__ theta, float* __restrict__ acc,
    const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ zoff,
    int Jb, int M, int nrow, int Nxy, int iz0, int nz, float Z0, float dZ, float t0, float dt,
    float rho, float slope, float z0, float beta0, float dsd, float dz, float v0, float dv,
    float theta_lo, float theta_hi, float t_max, float Qw, int K, float gamma_max, float reach,
    int n_off, float scale_out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nz * Nxy) return;
    int ixy = (int)(idx % Nxy), iz = iz0 + (int)(idx / Nxy);
    float x = X[ixy], y = Y[ixy], zv = Z0 + iz * dZ;
    float half = 0.5f * (nrow - 1);
    float zs[MAXP], sc[MAXP];
    bool ms[MAXP];
    float sum = 0.f;
    for (int j = 0; j < Jb; ++j) {
        float th = theta[j];
        float za = z0 + slope * (th - gamma_max - beta0), zb = z0 + slope * (th + gamma_max - beta0);
        if (zv < fminf(za, zb) - reach || zv > fmaxf(za, zb) + reach) continue;     // outside this view's cone
        float ti = 0.f;
        for (int kk = 0; kk <= 2 * K; ++kk) {
            float thk = th + (kk - K) * PI_F;
            float s = sinf(thk), c = cosf(thk);
            float tk = x * s - y * c;
            float bk = thk - asinf(fminf(fmaxf(tk / rho, -1.f), 1.f));
            float L = -(x - rho * cosf(bk)) * c - (y - rho * sinf(bk)) * s;
            zs[kk] = z0 + slope * (bk - beta0);
            sc[kk] = dsd / L;
            ms[kk] = (thk >= theta_lo) && (thk <= theta_hi) && (fabsf(tk) <= t_max);
            if (kk == K) ti = (tk - t0) / dt;
        }
        if (!ms[K] || ti <= -1.f || ti >= (float)M) continue;
        const float* Qj = Q + (long long)j * M * nrow;
        for (int o = 0; o < n_off; ++o) {
            float zc = zv + zoff[o];
            float total = 0.f, w0 = 0.f, r0 = 0.f;
            for (int kk = 0; kk <= 2 * K; ++kk) {
                if (!ms[kk]) continue;
                float row = ((zc - zs[kk]) * sc[kk] + dz - v0) / dv;
                float q = fabsf((row - half) / half);
                if (q >= 1.f) continue;
                float w = 1.f;
                if (q > Qw) { float cw = cosf(0.5f * PI_F * (q - Qw) / (1.f - Qw)); w = cw * cw; }
                total += w;
                if (kk == K) { w0 = w; r0 = row; }
            }
            if (w0 > 0.f) sum += w0 / fmaxf(total, 1e-6f) * bilinear(Qj, ti, r0, M, nrow);
        }
    }
    acc[(long long)iz * Nxy + ixy] += scale_out * sum;
}
'''

_KERNEL = None


def available(device) -> bool:
    """Whether the fused kernel can run on ``device``: a CUDA device, with CuPy installed."""
    if torch.device(device).type != 'cuda':
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy
        _KERNEL = cupy.RawKernel(_SOURCE, 'wfbp_bp')
    return _KERNEL


class _StreamHandle:
    """A raw CUDA stream, through the CUDA stream protocol (PyTorch's streams do not implement it yet)."""
    def __init__(self, handle: int):
        self.handle = handle

    def __cuda_stream__(self):
        return (0, self.handle)


def _torch_stream(device):
    """The current PyTorch stream on ``device``, as a CuPy stream, so that the kernels queue behind PyTorch's work."""
    import cupy
    handle = torch.cuda.current_stream(device).cuda_stream
    if hasattr(cupy.cuda.Stream, 'from_external'):       # CuPy 14 deprecates ExternalStream for it
        return cupy.cuda.Stream.from_external(_StreamHandle(handle))
    return cupy.cuda.ExternalStream(handle)


def backproject(Q: torch.Tensor, theta: np.ndarray, t: np.ndarray, dtheta: float, geo: dict, X: torch.Tensor, Y: torch.Tensor,
                Z: np.ndarray, out: torch.Tensor, Q_weight: float = 0.6, k_range: int | None = None, z_offsets=(0.0,),
                budget: float | None = None, views_per_launch: int = 128, theta_range: tuple | None = None) -> None:
    """Same as :func:`._wfbp.backproject` (and the same arguments), in one fused CUDA kernel. ``Z`` must be uniformly
    spaced."""
    import cupy
    device = out.device
    J, M, nrow = Q.shape
    Nx, Ny = X.shape
    Nxy, Nz = Nx * Ny, len(Z)
    Z = np.asarray(Z, dtype=np.float64)
    dZ = float(Z[1] - Z[0]) if Nz > 1 else 1.0
    if Nz > 1 and not np.allclose(np.diff(Z), dZ, rtol=1e-6, atol=1e-6):
        raise ValueError('the fused back projection needs uniformly spaced Z')
    X = X.to(device, torch.float32).reshape(-1).contiguous()
    Y = Y.to(device, torch.float32).reshape(-1).contiguous()
    rho, dsd, dz = geo['rho'], geo['dsd'], geo['dz']
    r_fov = float(torch.sqrt(X ** 2 + Y ** 2).max())
    if k_range is None:
        k_range = _wfbp.partner_range(geo, r_fov, theta, max(abs(o) for o in z_offsets))
    if k_range > MAX_K_RANGE:
        raise ValueError(f'the fused back projection holds at most {MAX_K_RANGE} partner rays either side')
    reach = (geo['n_rows'] / 2 + 1) * abs(geo['dv']) * (rho + r_fov) / dsd + 1.0 + max(abs(o) for o in z_offsets) + abs(dz)
    acc = torch.zeros((Nz, Nxy), dtype=torch.float32, device=device)
    zoff = torch.tensor(z_offsets, dtype=torch.float32, device=device)
    free = gpu_budget(budget, device) - 2 * out.numel() * 4
    Jb = int(max(1, min(views_per_launch, free // (M * nrow * 4 * 2))))
    t_max = rho * np.sin(geo['gamma_max'])
    first, last = theta_range if theta_range is not None else (theta[0], theta[-1])
    theta_lo, theta_hi = float(first) - 1e-6, float(last) + 1e-6
    f32, i32 = np.float32, np.int32
    kernel = _kernel()
    with _torch_stream(device):
        for s in range(0, J, Jb):
            e = min(J, s + Jb)
            th = theta[s:e]
            ends = geo['z0'] + geo['slope'] * (np.concatenate([th - geo['gamma_max'], th + geo['gamma_max']]) - geo['beta0'])
            k0 = int(np.searchsorted(Z, ends.min() - reach))
            k1 = int(np.searchsorted(Z, ends.max() + reach, side='right'))
            if k1 <= k0:
                continue
            Qb = Q[s:e].to(device, torch.float32).contiguous()
            thb = torch.tensor(th, dtype=torch.float32, device=device)
            n = (k1 - k0) * Nxy
            kernel(((n + 255) // 256,), (256,), (
                cupy.asarray(Qb), cupy.asarray(thb), cupy.asarray(acc), cupy.asarray(X), cupy.asarray(Y), cupy.asarray(zoff),
                i32(e - s), i32(M), i32(nrow), i32(Nxy), i32(k0), i32(k1 - k0), f32(Z[0]), f32(dZ), f32(t[0]), f32(t[1] - t[0]),
                f32(rho), f32(geo['slope']), f32(geo['z0']), f32(geo['beta0']), f32(dsd), f32(dz), f32(geo['v0']), f32(geo['dv']),
                f32(theta_lo), f32(theta_hi), f32(t_max), f32(Q_weight), i32(k_range), f32(geo['gamma_max']), f32(reach),
                i32(len(z_offsets)), f32(dtheta / len(z_offsets))))
            del Qb, thb
    out += acc.view(Nz, Nx, Ny).permute(1, 2, 0)
