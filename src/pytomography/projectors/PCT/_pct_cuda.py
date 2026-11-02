"""CUDA kernels of the proton CT projector (:mod:`.pct_system_matrix`), compiled at run time with CuPy (NVRTC), used
when CuPy is installed and the device is a CUDA GPU.

One thread follows one path: where its entry line first reaches the hull and its exit line, followed back, last
leaves it (a march on the hull's distance map), then the path itself (straight from the entry point to the hull, the
cubic spline inside it in ``segments`` pieces, straight to the exit point) voxel by voxel (Amanatides and Woo), with the
exact length in each voxel. Forward, it sums ``length * image`` and the length; back, it adds ``weight * residual *
length`` and ``weight * length`` to an interleaved pair of images (one 8-byte atomic per voxel on GPUs of compute
capability 9 or more). :mod:`.pct_system_matrix` has the same walk in PyTorch, for CPUs and GPUs without CuPy.

Coordinates are local to the grid (its lower corner at 0, in mm) and images are x fastest, ``[Lz, Ly, Lx]``.
"""
from __future__ import annotations
import numpy as np
import torch

_SOURCE = r'''
struct Grid { int n[3]; float dr[3]; };

// a straight piece a -> b through the grid: forward (sum of l x, length) or back projection into acc ([voxels][2])
__device__ void trace(const Grid& G, const float a[3], const float b[3], const float* __restrict__ x, bool back,
                      float r, float wgt, float& sum, float& len, float* acc) {
  float d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  float L = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (!(L > 1e-7f)) return;
  float t0 = 0.f, t1 = 1.f;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    float hi = G.n[k] * G.dr[k];
    if (fabsf(d[k]) < 1e-12f) { if (a[k] < 0.f || a[k] >= hi) return; continue; }
    float ta = (0.f - a[k]) / d[k], tb = (hi - a[k]) / d[k];
    t0 = fmaxf(t0, fminf(ta, tb)); t1 = fminf(t1, fmaxf(ta, tb));
  }
  if (t0 >= t1) return;
  int c[3], st[3];
  float tm[3], td[3];
  float tmid = 0.5f * (t0 + t1) - 0.5f * (t1 - t0) * 0.999f;     // just inside the entry
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    float pk = a[k] + tmid * d[k];
    c[k] = min(max((int)floorf(pk / G.dr[k]), 0), G.n[k] - 1);
    if (d[k] > 0.f) { st[k] = 1; tm[k] = ((c[k] + 1) * G.dr[k] - a[k]) / d[k]; td[k] = G.dr[k] / d[k]; }
    else if (d[k] < 0.f) { st[k] = -1; tm[k] = (c[k] * G.dr[k] - a[k]) / d[k]; td[k] = -G.dr[k] / d[k]; }
    else { st[k] = 0; tm[k] = 3e38f; td[k] = 3e38f; }
  }
  float t = t0;
  for (int guard = 0; guard < 4096 && t < t1; ++guard) {
    int k = (tm[0] <= tm[1] && tm[0] <= tm[2]) ? 0 : (tm[1] <= tm[2] ? 1 : 2);
    float tn = fminf(tm[k], t1);
    float seg = (tn - t) * L;
    if (seg > 0.f) {
      size_t idx = c[0] + (size_t)G.n[0] * (c[1] + (size_t)G.n[1] * c[2]);
      if (back) {
#if __CUDA_ARCH__ >= 900
        atomicAdd(reinterpret_cast<float2*>(acc) + idx, make_float2(seg * r * wgt, seg * wgt));
#else
        atomicAdd(acc + 2 * idx, seg * r * wgt); atomicAdd(acc + 2 * idx + 1, seg * wgt);
#endif
      }
      else { sum += seg * __ldg(x + idx); len += seg; }
    }
    t = tn;
    if (tn >= t1) break;
    c[k] += st[k];
    if (c[k] < 0 || c[k] >= G.n[k]) break;
    tm[k] += td[k];
  }
}

// how far along p + s d (s >= 0) the line first reaches the hull (distance map dist, 0 inside), and whether it does:
// from where it enters the grid, steps of the distance less two voxel half-diagonals (at least half a voxel); with no
// distance map the hull is the whole grid
__device__ bool first_reach(const Grid& G, const float p[3], const float d[3], const float* __restrict__ dist,
                            float& s_out) {
  float near = -3e38f, far = 3e38f;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    float dk = fabsf(d[k]) < 1e-12f ? 1e-12f : d[k];
    float t1 = (0.f - p[k]) / dk, t2 = (G.n[k] * G.dr[k] - p[k]) / dk;
    near = fmaxf(near, fminf(t1, t2)); far = fminf(far, fmaxf(t1, t2));
  }
  near = fmaxf(near, 0.f);
  if (!(far > near)) return false;
  if (!dist) { s_out = near; return true; }
  float margin = sqrtf(G.dr[0] * G.dr[0] + G.dr[1] * G.dr[1] + G.dr[2] * G.dr[2]);
  float step = 0.5f * fminf(G.dr[0], fminf(G.dr[1], G.dr[2]));
  float s = near;
  for (int it = 0; it < 512; ++it) {
    int v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) v[k] = min(max((int)floorf((p[k] + s * d[k]) / G.dr[k]), 0), G.n[k] - 1);
    float dv = __ldg(dist + v[0] + (size_t)G.n[0] * (v[1] + (size_t)G.n[1] * v[2]));
    if (dv <= 0.f) { s_out = s; return true; }
    s += fmaxf(dv - margin, step);
    if (!(s < far)) return false;
  }
  return false;
}

__device__ __forceinline__ bool hull_crossings(const Grid& G, const float a[3], const float da[3], const float b[3],
                                               const float db[3], const float* __restrict__ dist, float g0[3],
                                               float g1[3]) {
  float s0, s1, back_d[3] = {-db[0], -db[1], -db[2]};
  bool hit = first_reach(G, a, da, dist, s0) & first_reach(G, b, back_d, dist, s1);
  if (!hit) return false;
#pragma unroll
  for (int k = 0; k < 3; ++k) { g0[k] = a[k] + s0 * da[k]; g1[k] = b[k] - s1 * db[k]; }
  return true;
}

__device__ __forceinline__ void load_path(int i, const float* p_in, const float* d_in, const float* p_out,
                                          const float* d_out, const float lo[3], float a[3], float da[3], float b[3],
                                          float db[3]) {
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    a[k] = p_in[3 * i + k] - lo[k]; b[k] = p_out[3 * i + k] - lo[k];
    da[k] = d_in[3 * i + k]; db[k] = d_out[3 * i + k];
  }
}

// forward (back 0): fwd[i] = sum of l x along path i, len[i] = its length in the grid; back (back 1): each path adds
// w[i] r l and w[i] l to acc, r = rin[i / K] (K paths per proton), skipping paths of zero length when len is given
extern "C" __global__ void __launch_bounds__(128) project(
    const float* __restrict__ x, int n, const float* __restrict__ p_in, const float* __restrict__ d_in,
    const float* __restrict__ p_out, const float* __restrict__ d_out, const float* __restrict__ dist, int has_dist,
    int segments, const float* __restrict__ w, const float* __restrict__ rin, int K, int back, int has_len,
    float* fwd, float* len, float* acc, int nx, int ny, int nz, float drx, float dry, float drz, float lox,
    float loy, float loz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Grid G = {{nx, ny, nz}, {drx, dry, drz}};
  const float lo[3] = {lox, loy, loz};
  float a[3], b[3], g0[3], g1[3], da[3], db[3];
  load_path(i, p_in, d_in, p_out, d_out, lo, a, da, b, db);
  bool bk = back != 0;
  float sum = 0.f, length = 0.f, r = 0.f;
  const float wi = bk ? w[i] : 1.f;
  if (bk) {
    if (has_len && !(len[i] > 0.f)) return;
    r = rin[i / K];
  }
  if (!hull_crossings(G, a, da, b, db, has_dist ? dist : nullptr, g0, g1)) {
    trace(G, a, b, x, bk, r, wi, sum, length, acc);
  } else {
    float chord = sqrtf((g1[0] - g0[0]) * (g1[0] - g0[0]) + (g1[1] - g0[1]) * (g1[1] - g0[1]) +
                        (g1[2] - g0[2]) * (g1[2] - g0[2]));
    trace(G, a, g0, x, bk, r, wi, sum, length, acc);
    float q[3] = {g0[0], g0[1], g0[2]};
    for (int s = 1; s <= segments; ++s) {
      float t = (float)s / segments, t2 = t * t, t3 = t2 * t;
      float c0 = 2 * t3 - 3 * t2 + 1, c1 = (t3 - 2 * t2 + t) * chord, c2 = -2 * t3 + 3 * t2, c3 = (t3 - t2) * chord;
      float qn[3];
#pragma unroll
      for (int k = 0; k < 3; ++k) qn[k] = c0 * g0[k] + c1 * da[k] + c2 * g1[k] + c3 * db[k];
      trace(G, q, qn, x, bk, r, wi, sum, length, acc);
      q[0] = qn[0]; q[1] = qn[1]; q[2] = qn[2];
    }
    trace(G, g1, b, x, bk, r, wi, sum, length, acc);
  }
  if (!bk) { fwd[i] = sum; len[i] = length; }
}

// the hull crossings alone, in the object frame (hit 0 or 1), for binning by the hull chord
extern "C" __global__ void hull_points(int n, const float* p_in, const float* d_in, const float* p_out,
                                       const float* d_out, const float* dist, int has_dist, float* h0, float* h1,
                                       unsigned char* hit, int nx, int ny, int nz, float drx, float dry, float drz,
                                       float lox, float loy, float loz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const Grid G = {{nx, ny, nz}, {drx, dry, drz}};
  const float lo[3] = {lox, loy, loz};
  float a[3], b[3], da[3], db[3], g0[3] = {0, 0, 0}, g1[3] = {0, 0, 0};
  load_path(i, p_in, d_in, p_out, d_out, lo, a, da, b, db);
  bool ok = hull_crossings(G, a, da, b, db, has_dist ? dist : nullptr, g0, g1);
#pragma unroll
  for (int k = 0; k < 3; ++k) { h0[3 * i + k] = g0[k] + lo[k]; h1[3 * i + k] = g1[k] + lo[k]; }
  hit[i] = ok ? 1 : 0;
}
'''

_MODULE = None


def available(device) -> bool:
    """Whether the kernels can run on ``device``: a CUDA device, with CuPy installed."""
    if torch.device(device).type != 'cuda':
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _kernel(name: str):
    global _MODULE
    if _MODULE is None:
        import cupy
        # fast math, as the prototype the published images came from (nvcc -use_fast_math)
        _MODULE = cupy.RawModule(code=_SOURCE, options=('--use_fast_math',))
    return _MODULE.get_function(name)


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


def _grid(shape, dr, lo) -> tuple:
    return (*(np.int32(s) for s in shape), *(np.float32(v) for v in dr), *(np.float32(v) for v in lo))


def project(image_x, paths: dict, distance, segments: int, shape, dr, lo, residual=None, K: int = 1, length=None,
            acc=None):
    """Forward (``residual`` None): each path's sum of length times ``image_x`` and its length in the grid, two [n]
    tensors. Back: adds ``weight * residual[i // K] * length`` and ``weight * length`` along path ``i`` into ``acc``
    ([voxels, 2], x fastest), skipping the paths whose forward ``length`` is 0 when it is given; returns None.
    ``paths``: ``entry_points``, ``entry_directions``, ``exit_points``, ``exit_directions`` [n, 3] and ``weights`` [n]
    (object frame, mm), all float32 on the device; ``distance``: the hull's distance map, x fastest, or None."""
    import cupy
    p = [paths[k] for k in ('entry_points', 'entry_directions', 'exit_points', 'exit_directions')]
    n = p[0].shape[0]
    device = p[0].device
    back = residual is not None
    dummy = torch.zeros(1, dtype=torch.float32, device=device)
    fwd = dummy if back else torch.empty(n, dtype=torch.float32, device=device)
    out_len = (length if length is not None else dummy) if back else torch.empty(n, dtype=torch.float32, device=device)
    if n:
        args = (cupy.asarray(image_x if image_x is not None else dummy), np.int32(n), *(cupy.asarray(a) for a in p),
                cupy.asarray(distance if distance is not None else dummy), np.int32(distance is not None),
                np.int32(segments), cupy.asarray(paths['weights']), cupy.asarray(residual if back else dummy),
                np.int32(K), np.int32(back), np.int32(back and length is not None), cupy.asarray(fwd),
                cupy.asarray(out_len), cupy.asarray(acc if back else dummy), *_grid(shape, dr, lo))
        with _torch_stream(device):
            _kernel('project')(((n + 127) // 128,), (128,), args)
    return None if back else (fwd, out_len)


def hull_points(paths: dict, distance, shape, dr, lo):
    """Where each path's entry line first reaches the hull and its exit line last leaves it (object frame, [n, 3], mm),
    and whether both do ([n] bool); as :func:`.pct_system_matrix.hull_points`."""
    import cupy
    p = [paths[k] for k in ('entry_points', 'entry_directions', 'exit_points', 'exit_directions')]
    n = p[0].shape[0]
    device = p[0].device
    h0 = torch.empty((n, 3), dtype=torch.float32, device=device)
    h1 = torch.empty_like(h0)
    hit = torch.empty(n, dtype=torch.uint8, device=device)
    if n:
        dummy = torch.zeros(1, dtype=torch.float32, device=device)
        with _torch_stream(device):
            _kernel('hull_points')(((n + 127) // 128,), (128,), (
                np.int32(n), *(cupy.asarray(a) for a in p), cupy.asarray(distance if distance is not None else dummy),
                np.int32(distance is not None), cupy.asarray(h0), cupy.asarray(h1), cupy.asarray(hit),
                *_grid(shape, dr, lo)))
    return h0, h1, hit.bool()
