# Copyright (c) 2026 H. Wei
# SPDX-License-Identifier: MIT
# See the LICENSE file in the repository root for license terms.

"""Kernel expectation maximization (KEM) with a precomputed sparse kernel, and the two ways this module builds one.

* :class:`ExternalKEMTransform`: a kernel given as each voxel's ``k`` neighbours and their weights, for
  :class:`~pytomography.projectors.shared.KEMSystemMatrix` and :class:`~pytomography.algorithms.KEM`.
* :func:`construct_kernel`: the k nearest neighbours of each voxel in a prior image (PET, MR...), by its intensity or
  patches and by distance.
* :class:`UNet3D`, :func:`train_deep_kernel` and :func:`construct_deep_kernel`: the PET-informed deep-learned kernel
  (H. Wei, L. Livieratos and A. J. Reader, "PET-Informed Deep-Learned Kernel Method for Theranostic SPECT
  Reconstruction", IEEE TRPMS 2026, doi:10.1109/TRPMS.2026.3737413): a U-Net learns features of the prior, and the
  neighbours are found in those features.
"""
from __future__ import annotations

import itertools
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

import pytomography
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.transforms import Transform


class ExternalKEMTransform(Transform):
    r"""Load a precomputed kernel matrix weights and neighbor indices from npy files, and 
    do forward and backward operations for kernel expectation maximization (KEM).

    Use this with ``KEMSystemMatrix(system_matrix, transform)`` and ``KEM``.
    The system matrix configures the transform's image shape automatically.
    This operator is fixed during reconstruction.

    Args:
        weights (numpy.ndarray | torch.Tensor): Records weights for neighbors.
        indices (numpy.ndarray | torch.Tensor): Records the neighbor indices for aggregation.
        kernel_on_gpu (bool): Store the full kernel on the PyTomography
            compute device. Otherwise keep it on CPU and transfer chunks.
        chunk_size (int): Maximum number of rows processed at once.
    """

    def __init__(self, weights, indices, kernel_on_gpu: bool = False,
                 chunk_size: int = 65536) -> None:
        super().__init__()
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
            raise TypeError('chunk_size must be a positive integer')
        if chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer')
        weights = torch.as_tensor(weights)
        indices = torch.as_tensor(indices)
        if weights.ndim != 2 or not all(weights.shape):
            raise ValueError('weights must have nonempty shape (N_voxels, k)')
        if indices.shape != weights.shape:
            raise ValueError('weights and indices must have the same shape')
        if weights.is_complex() or weights.dtype == torch.bool:
            raise TypeError('weights must be real numeric values')
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16,
                                 torch.int32, torch.int64):
            raise TypeError('indices must have integer dtype')
        self.device = torch.device(pytomography.device)
        self.dtype = pytomography.dtype
        self.kernel_on_gpu = kernel_on_gpu
        self.chunk_size = int(chunk_size)
        storage_device = self.device if kernel_on_gpu else torch.device('cpu')
        self.kernel = weights.detach().to(device=storage_device, dtype=self.dtype).clone()
        self.indices = indices.detach().to(device=storage_device, dtype=torch.long).clone()
        self.n_voxels = self.kernel.shape[0]
        for start in range(0, self.n_voxels, self.chunk_size):
            w = self.kernel[start:start + self.chunk_size]
            idx = self.indices[start:start + self.chunk_size]
            if not torch.isfinite(w).all() or (w < 0).any():
                raise ValueError('weights must be finite and nonnegative')
            if not (w > 0).any(dim=1).all():
                raise ValueError('every kernel row must have positive weight')
            if (idx < 0).any() or (idx >= self.n_voxels).any():
                raise ValueError('indices must lie in [0, N_voxels)')
        self._shape = None

    @classmethod
    def from_files(cls, kernel_path: str | Path, indices_path: str | Path,
                   **kwargs) -> ExternalKEMTransform:
        return cls(np.load(kernel_path, allow_pickle=False),
                   np.load(indices_path, allow_pickle=False), **kwargs)

    def configure(self, object_meta: ObjectMeta, proj_meta: ProjMeta) -> None:
        shape = tuple(object_meta.shape)
        if len(shape) != 3 or np.prod(shape) != self.n_voxels:
            raise ValueError('object metadata must describe a 3D grid with N_voxels entries')
        super().configure(object_meta, proj_meta)
        self._shape = shape

    def _flatten(self, object: torch.Tensor) -> torch.Tensor:
        if self._shape is None:
            raise RuntimeError('configure the transform before applying it')
        if tuple(object.shape) != self._shape:
            raise ValueError(f'expected image shape {self._shape}, got {tuple(object.shape)}')
        return object.to(device=self.device, dtype=self.dtype).reshape(-1)

    def _chunks(self):
        for start in range(0, self.n_voxels, self.chunk_size):
            end = min(start + self.chunk_size, self.n_voxels)
            yield (start, end, self.kernel[start:end].to(self.device),
                   self.indices[start:end].to(self.device))

    @torch.no_grad()
    def forward(self, object: torch.Tensor) -> torch.Tensor:
        """Map coefficient image to activity image with :math:`K`."""
        flat = self._flatten(object)
        result = torch.empty_like(flat)
        for start, end, weights, indices in self._chunks():
            result[start:end] = (weights * flat[indices]).sum(dim=1)
        return result.reshape(self._shape)

    @torch.no_grad()
    def backward(self, object: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`K^T` by accumulating all incoming neighbor weights."""
        flat = self._flatten(object)
        result = torch.zeros_like(flat)
        for start, end, weights, indices in self._chunks():
            result.index_add_(0, indices.reshape(-1),
                              (weights * flat[start:end, None]).reshape(-1))
        return result.reshape(self._shape)


# --- k-nearest-neighbour kernels from a prior image

def _positive_integer(value, name, odd=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be a positive integer')
    if value <= 0 or (odd and value % 2 == 0):
        raise ValueError(f'{name} must be positive' + (' and odd' if odd else ''))
    return int(value)


def construct_kernel(prior, k: int = 30, sigmaf: float = 1.,
                     sigmac: float = 1., patch: int = 3,
                     local_window: int = 11, mode: str = 'local', *,
                     include_self: bool = False, chunk_size: int = 256,
                     device: str | torch.device = 'cpu'):
    r"""Build a row-normalized kernel from prior image.

        E_{ij} = \frac{\|v_i-v_j\|_2^2}{2\sigma_f^2}
                 + \frac{\|r_i-r_j\|_2^2}{2\sigma_c^2}.

    Args:
        prior (numpy.ndarray): prior image array (PET, MR...) Feature preparation uses NumPy; local neighbor search can use CUDA.
        k (int): Number of k-nearest neighbors. Must not exceed the maximum
            available candidates (excluding self unless requested).
        sigmaf (float): Feature bandwidth in standardized-feature
            units. Smaller values penalize feature differences more strongly.
        sigmac (float): Coordinate bandwidth in voxel units.
            Smaller values favor nearby voxels. Assumes isotropic voxels.
        patch (int): Odd side length of the cubic feature patch. ``1`` uses
            voxel intensity; ``3`` uses 27 components. 
        local_window (int): Neighbor searching local window.
            ``mode='local'``. Ignored in global mode.
        mode (str): ``'local'`` searches in-volume neighbors in the cube;
            ``'global'`` searches all voxels using a KD-tree in joint
            feature/coordinate space. Both use the same energy and weights.
        include_self (bool): Include the center voxel as a candidate. Defaults
            to False.
        chunk_size (int): Number of query voxels per batch. Increase this to
            reduce batch overhead, subject to available RAM or GPU memory.
        device (str | torch.device): ``'cpu'`` (default) keeps the existing
            NumPy/SciPy implementation. ``'cuda'`` or ``'cuda:0'`` accelerates
            local distances, stable neighbor selection and normalization with
            PyTorch. Global mode only supports CPU. CUDA uses float64 distance
            arithmetic to preserve CPU behavior; outputs remain NumPy arrays.
            Feature preparation is on CPU. GPU memory scales with the feature
            array and ``chunk_size * local_window**3``, not all candidate pairs.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(weights, indices)`` of shape
        ``(prior.size, k)``, with float32 weights and int64 C-order flat voxel
        indices, ready for ``ExternalKEMTransform(weights, indices)`` or
        ``numpy.save``. Each row sums to one. At local boundaries with fewer
        than k candidates, unused slots have zero weight and a valid index.

    Example:
        >>> weights, indices = construct_kernel(prior, k=30, patch=3,
        ...     sigmaf=1.0, sigmac=1.0, local_window=11, mode='local')
        >>> kernel = ExternalKEMTransform(weights, indices)
    """
    k = _positive_integer(k, 'k')
    patch = _positive_integer(patch, 'patch', odd=True)
    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    if mode not in ('local', 'global'):
        raise ValueError("mode must be 'local' or 'global'")
    if not isinstance(include_self, bool):
        raise TypeError('include_self must be a bool')
    for name, value in (('sigmaf', sigmaf), ('sigmac', sigmac)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'{name} must be a positive finite number')
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    prior = np.asarray(prior)
    if prior.ndim != 3 or not all(prior.shape):
        raise ValueError('prior must be a nonempty 3D image')
    if prior.dtype.kind not in 'iuf':
        raise TypeError('prior must contain real numeric values')
    prior = prior.astype(np.float64)
    if not np.isfinite(prior).all():
        raise ValueError('prior must contain only finite values')
    shape = np.asarray(prior.shape)
    n = prior.size
    max_candidates = n - (not include_self)
    if mode == 'local':
        local_window = _positive_integer(local_window, 'local_window', odd=True)
        max_candidates = int(np.prod(np.minimum(shape, local_window))) - (not include_self)
    if k > max_candidates:
        raise ValueError(f'k exceeds the maximum {max_candidates} available neighbors')

    compute_device = torch.device(device)
    if compute_device.type not in ('cpu', 'cuda'):
        raise ValueError("device must be 'cpu' or 'cuda' (optionally with an index)")
    if compute_device.type == 'cuda':
        if mode != 'local':
            raise ValueError("CUDA kernel construction only supports mode='local'; use device='cpu' for global mode")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='cpu' or a CUDA-enabled environment")
        return _construct_local_kernel_torch(prior, k=k, sigmaf=sigmaf,
            sigmac=sigmac, patch=patch, local_window=local_window,
            include_self=include_self, chunk_size=chunk_size, device=compute_device)

    padded = np.pad(prior, patch // 2, mode='reflect')
    features = np.lib.stride_tricks.sliding_window_view(padded, (patch,)*3).reshape(n, -1).copy()
    feature_std = features.std() + 1e-8
    # Subtract a common mean to improve conditioning without changing distances.
    features -= features.mean()
    features /= feature_std
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        features /= sigmaf
        coords = np.column_stack(np.unravel_index(np.arange(n), prior.shape))
        scaled_coords = coords / sigmac
    if not np.isfinite(features).all() or not np.isfinite(scaled_coords).all():
        raise ValueError('prior or bandwidths exceed numerical range; rescale inputs')
    weights = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)
    if mode == 'global':
        augmented = np.column_stack((features, scaled_coords))
        tree = cKDTree(augmented)
    else:
        radius = np.minimum(local_window // 2, shape - 1)
        offsets = np.stack(np.meshgrid(*(np.arange(-r, r+1) for r in radius),
                                      indexing='ij'), axis=-1).reshape(-1, 3)
        if not include_self:
            offsets = offsets[(offsets != 0).any(axis=1)]
        spatial = np.sum((offsets / sigmac)**2, axis=1)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        rows = np.arange(start, end)
        if mode == 'global':
            distances, candidate_indices = tree.query(augmented[start:end],
                k=list(range(1, k + (not include_self) + 1)))
            energy = distances**2
            if not include_self:
                energy[candidate_indices == rows[:, None]] = np.inf
        else:
            neighbors = coords[start:end, None, :] + offsets
            valid = ((neighbors >= 0) & (neighbors < shape)).all(axis=2)
            safe = np.clip(neighbors, 0, shape - 1)
            candidate_indices = np.ravel_multi_index(tuple(safe.transpose(2, 0, 1)), prior.shape)
            energy = np.broadcast_to(spatial, candidate_indices.shape).copy()
            # Avoid a B x window**3 x patch**3 temporary feature tensor.
            for component in range(features.shape[1]):
                difference = features[candidate_indices, component] - features[start:end, component, None]
                energy += difference**2
            energy[~valid] = np.inf
        selected = np.argsort(energy, axis=1, kind='stable')[:, :k]
        selected_energy = np.take_along_axis(energy, selected, axis=1)
        selected_indices = np.take_along_axis(candidate_indices, selected, axis=1)
        minimum = selected_energy[:, :1]
        if not np.isfinite(minimum).all():
            raise ValueError('no finite neighbor energy; increase bandwidths or rescale prior')
        row_weights = np.exp(-0.5 * (selected_energy - minimum))
        row_weights /= row_weights.sum(axis=1, keepdims=True)
        selected_indices[~np.isfinite(selected_energy)] = 0
        indices[start:end] = selected_indices
        weights[start:end] = row_weights
    return weights, indices


@torch.no_grad()
def _construct_local_kernel_torch(prior, *, k, sigmaf, sigmac, patch,
                                  local_window, include_self, chunk_size, device):
    """Tensor backend for validated local inputs; also executable on CPU."""
    # Keep feature standardization identical to NumPy, including reflected
    # patches on singleton axes. Transfer once, then search entirely on device.
    shape = np.asarray(prior.shape)
    n = prior.size
    padded = np.pad(prior.astype(np.float64), patch // 2, mode='reflect')
    features_np = np.lib.stride_tricks.sliding_window_view(padded, (patch,)*3).reshape(n, -1).copy()
    feature_std = features_np.std() + 1e-8
    features_np -= features_np.mean()
    features_np /= feature_std
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        features_np /= sigmaf
        coords_np = np.column_stack(np.unravel_index(np.arange(n), prior.shape))
        scaled_coords = coords_np / sigmac
    if not np.isfinite(features_np).all() or not np.isfinite(scaled_coords).all():
        raise ValueError('prior or bandwidths exceed numerical range; rescale inputs')
    features = torch.as_tensor(features_np, device=device)
    coords = torch.as_tensor(coords_np, device=device)
    del features_np, coords_np, scaled_coords, padded
    shape_t = torch.as_tensor(shape, device=device)
    radius = np.minimum(local_window // 2, shape - 1)
    offsets_np = np.stack(np.meshgrid(*(np.arange(-r, r+1) for r in radius),
                                      indexing='ij'), axis=-1).reshape(-1, 3)
    if not include_self:
        offsets_np = offsets_np[(offsets_np != 0).any(axis=1)]
    # Identical offset ordering and stable sorting retain the CPU tie rule.
    spatial = torch.as_tensor(np.sum((offsets_np / sigmac)**2, axis=1), device=device)
    offsets = torch.as_tensor(offsets_np, device=device)
    weights = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        neighbors = coords[start:end, None, :] + offsets
        valid = ((neighbors >= 0) & (neighbors < shape_t)).all(dim=2)
        safe = torch.minimum(neighbors.clamp_min(0), shape_t - 1)
        candidate_indices = (safe[..., 0] * int(shape[1]) + safe[..., 1]) * int(shape[2]) + safe[..., 2]
        del neighbors, safe
        energy = spatial.expand(end-start, -1).clone()
        # Avoid a batch x candidates x patch-components tensor.
        for component in range(features.shape[1]):
            difference = features[candidate_indices, component] - features[start:end, component, None]
            energy += difference.square()
        energy.masked_fill_(~valid, torch.inf)
        selected = torch.argsort(energy, dim=1, stable=True)[:, :k]
        selected_energy = energy.gather(1, selected)
        selected_indices = candidate_indices.gather(1, selected)
        minimum = selected_energy[:, :1]
        if not torch.isfinite(minimum).all().item():
            raise ValueError('no finite neighbor energy; increase bandwidths or rescale prior')
        row_weights = torch.exp(-0.5 * (selected_energy - minimum))
        row_weights /= row_weights.sum(dim=1, keepdim=True)
        selected_indices.masked_fill_(~torch.isfinite(selected_energy), 0)
        weights[start:end] = row_weights.to(dtype=torch.float32).cpu().numpy()
        indices[start:end] = selected_indices.cpu().numpy()
    return weights, indices


# --- the deep-learned kernel: its U-Net, its training and its kernel

def _conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.BatchNorm3d(out_channels), nn.LeakyReLU(.01, inplace=False),
    )


def _double_conv(in_channels, out_channels):
    return nn.Sequential(_conv(in_channels, out_channels), _conv(out_channels, out_channels))


class UNet3D(nn.Module):
    """Three-level 3D U-Net with additive skips and nonnegative features.

    Args:
        in_channels (int): Number of input channels, normally 1 for a prior.
        base_channels (int): Encoder width; successive levels use 1, 2, 4,
            and 8 times this width.
        feature_channels (int): Number of learned features per voxel.

    Inputs and outputs follow [batch, channels, X, Y, Z] order. Spatial sizes
    must be at least 16; odd sizes are supported. The architecture adapts the
    user's deep_kem.py with fixed-scale interpolation and high-side cropping
    to maintain coordinates during aligned tiled inference.
    """
    def __init__(self, in_channels=1, base_channels=16, feature_channels=16):
        super().__init__()
        for name, value in [('in_channels', in_channels), ('base_channels', base_channels),
                            ('feature_channels', feature_channels)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        channels = [base_channels * 2**level for level in range(4)]
        self.encoders = nn.ModuleList([_double_conv(in_channels, channels[0])] +
            [_double_conv(channels[i-1], channels[i]) for i in range(1, 4)])
        self.down = nn.ModuleList([_conv(c, c, stride=2) for c in channels[:3]])
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv3d(channels[i+1], channels[i], 1),
                          nn.BatchNorm3d(channels[i]), nn.LeakyReLU(.01))
            for i in range(3)
        ])
        self.decoders = nn.ModuleList([_double_conv(c, c) for c in channels[:3]])
        self.output = nn.Sequential(nn.Conv3d(channels[0], feature_channels, 3, padding=1),
                                    nn.BatchNorm3d(feature_channels), nn.ReLU())
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, a=.01, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, value):
        if value.ndim != 5 or value.shape[1] != self.in_channels or min(value.shape[2:]) < 16:
            raise ValueError('Expected [B, in_channels, X, Y, Z] with spatial sizes >= 16')
        skips = []
        for level in range(4):
            value = self.encoders[level](value)
            if level < 3:
                skips.append(value)
                value = self.down[level](value)
        for level in (2, 1, 0):
            value = self.up[level](F.interpolate(value, scale_factor=2,
                mode='trilinear', align_corners=False))
            shape = skips[level].shape[2:]
            value = value[:, :, :shape[0], :shape[1], :shape[2]]
            value = self.decoders[level](value + skips[level])
        return self.output(value)


def _validate_neighbors(shape, k, local_window, include_self):
    k = _positive_integer(k, 'k')
    local_window = _positive_integer(local_window, 'local_window', odd=True)
    if not isinstance(include_self, bool):
        raise TypeError('include_self must be a bool')
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('Expected a nonempty 3D shape')
    maximum = int(np.prod(np.minimum(shape, local_window))) - (not include_self)
    if k > maximum:
        raise ValueError(f'k exceeds the maximum {maximum} available neighbors')


def _normalize_image(image, name):
    array = np.asarray(image)
    if array.ndim != 3 or array.dtype.kind not in 'iuf' or not all(array.shape):
        raise ValueError(f'{name} must be a nonempty real 3D image')
    array = np.array(array, dtype=np.float32, copy=True, order='C')
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f'{name} must be finite and nonnegative')
    array -= array.min()
    maximum = float(array.max())
    if maximum > 0:
        array /= maximum
    return array


def _model_device(model, device):
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if device.type not in ('cpu', 'cuda'):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use device='cpu'")
    return device


def _deep_kernel_rows(features, shape, rows, *, k=50, local_window=9,
                      include_self=False):
    """Differentiable valid local kNN rows; weights are softmax(-distance)."""
    _validate_neighbors(shape, k, local_window, include_self)
    if features.ndim != 2 or features.shape[0] != int(np.prod(shape)):
        raise ValueError('Features must have shape [prod(shape), channels]')
    if rows.ndim != 1 or not rows.numel() or rows.min() < 0 or rows.max() >= features.shape[0]:
        raise ValueError('Invalid query rows')
    radius = np.minimum(local_window // 2, np.asarray(shape)-1)
    axes = [torch.arange(-int(r), int(r)+1, device=features.device) for r in radius]
    offsets = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    if not include_self:
        offsets = offsets[offsets.ne(0).any(-1)]
    coords = torch.stack((rows // (shape[1]*shape[2]),
                          rows // shape[2] % shape[1], rows % shape[2]), 1)
    neighbors = coords[:, None] + offsets
    valid = ((neighbors >= 0) & (neighbors < torch.tensor(shape, device=features.device))).all(-1)
    indices = (neighbors[..., 0]*shape[1] + neighbors[..., 1])*shape[2] + neighbors[..., 2]
    indices = torch.where(valid, indices, rows[:, None])
    distance = torch.linalg.vector_norm(features[indices] - features[rows, None], dim=-1)
    distance = distance.masked_fill(~valid, torch.inf)
    selected = torch.argsort(distance, dim=1, stable=True)[:, :k]
    best = distance.gather(1, selected)
    weights = torch.softmax(-(best - best[:, :1]), dim=1)
    indices = indices.gather(1, selected)
    indices = indices.masked_fill(~torch.isfinite(best), 0)
    return weights, indices


def train_deep_kernel(model, prior, coefficient_image, target=None, *,
                      epochs=20, patches_per_epoch=200, patch_size=32,
                      chunk_size=512, k=50, local_window=9,
                      include_self=False, learning_rate=1e-3, device=None,
                      seed=2026, verbose=True):
    """Train a feature U-Net from aligned image patches using cross-modal MSE.

    Args:
        model (torch.nn.Module): Feature network, e.g. UNet3D. Updated in place.
        prior (numpy.ndarray): Aligned PET prior, used as the network input.
        coefficient_image (numpy.ndarray): Image aggregated by the learned
            kernel, e.g. the previously reconstructed SPECT MLEM image.
        target (numpy.ndarray | None): Image used as the MSE target; defaults
            to prior. Each input is independently min-max normalized to [0,1].
        epochs (int): Number of training epochs.
        patches_per_epoch (int): Number of random patches/Adam updates per epoch.
        patch_size (int): Cubic crop size, a multiple of 8 and >=16.
        chunk_size (int): Number of center voxels processed per batch. Every
            patch voxel contributes to loss; this controls memory, not sampling.
        k (int): Retained neighbors per query. Use the same value for export.
        local_window (int): Odd local search width, smaller than patch_size.
        include_self (bool): Whether to include a voxel's own feature.
        learning_rate (float): Adam learning rate.
        device (str | torch.device | None): CPU/CUDA; defaults to model device.
        seed (int): Seed for random crops and training operations.
            Initialize the model after setting torch.manual_seed for repeatable
            initial weights. CUDA may still have nondeterministic operations.
        verbose (bool): Print mean loss after each epoch.

    Returns:
        list[float]: Mean MSE per epoch. Model is returned to eval mode.

    The objective is MSE(K_theta(prior) @ coefficient_image, target), alternatively you can change
    the objective function to negative normalized mutual information like described in H. Wei, L. Livieratos, and A. J. Reader, “PET-Informed Deep-Learned Kernel Method for Theranostic SPECT Reconstruction,” IEEE Transactions on Radiation and Plasma Medical Sciences, early access, 2026, doi: 10.1109/TRPMS.2026.3737413.https://ieeexplore.ieee.org/document/11707312.
    """
    for name, value in [('epochs', epochs), ('patches_per_epoch', patches_per_epoch),
                        ('patch_size', patch_size), ('chunk_size', chunk_size)]:
        _positive_integer(value, name)
    if patch_size < 16 or patch_size % 8:
        raise ValueError('patch_size must be a multiple of 8 and >=16')
    _validate_neighbors((patch_size,)*3, k, local_window, include_self)
    if local_window >= patch_size:
        raise ValueError('local_window must be smaller than patch_size')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be positive and finite')
    images = [_normalize_image(a, name) for a, name in
        [(prior, 'prior'), (coefficient_image, 'coefficient_image'),
         (prior if target is None else target, 'target')]]
    shape = images[0].shape
    if any(a.shape != shape for a in images) or min(shape) < patch_size:
        raise ValueError('Images must match and contain a complete training patch')
    device = _model_device(model, device)
    model.to(device=device, dtype=torch.float32).train()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        total = 0.
        for _ in range(patches_per_epoch):
            start = [int(rng.integers(0, size-patch_size+1)) for size in shape]
            region = tuple(slice(s, s+patch_size) for s in start)
            p, guide, reference = [torch.from_numpy(a[region].copy()).to(device) for a in images]
            optimizer.zero_grad(set_to_none=True)
            feature_map = model(p[None, None])
            if feature_map.ndim != 5 or feature_map.shape[0] != 1 or tuple(feature_map.shape[2:]) != (patch_size,)*3:
                raise ValueError('Model must return [1, feature_channels, X, Y, Z]')
            features = feature_map[0].permute(1,2,3,0).reshape(patch_size**3,-1)
            # Detach only at the feature boundary, accumulate exact dL/df,
            # then propagate that gradient through the U-Net once per patch.
            # Each local-row graph is freed after its backward pass, keeping
            # kernel activation memory bounded independently of patch size.
            feature_leaf = features.detach().requires_grad_(True)
            n = patch_size**3
            guide_flat, target_flat = guide.flatten(), reference.flatten()
            patch_loss = 0.
            for start_row in range(0, n, chunk_size):
                stop_row = min(n, start_row + chunk_size)
                rows = torch.arange(start_row, stop_row, device=device)
                weights, indices = _deep_kernel_rows(feature_leaf, (patch_size,)*3, rows,
                    k=k, local_window=local_window, include_self=include_self)
                prediction = (weights * guide_flat[indices]).sum(1)
                # Normalize by the entire patch, including an unequal final
                # chunk, so this is the full-patch mean rather than a sum of means.
                loss = (prediction - target_flat[rows]).square().sum() / n
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite deep-kernel training loss')
                loss.backward()
                patch_loss += loss.detach().item()
            features.backward(feature_leaf.grad)
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise FloatingPointError('Nonfinite deep-kernel gradient')
            optimizer.step()
            total += patch_loss
        history.append(total / patches_per_epoch)
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} | MSE: {history[-1]:.6f}', flush=True)
    model.eval()
    return history


@torch.no_grad()
def _infer_deep_features(model, prior, device, inference_patch_size):
    """Evaluate full or aligned halo tiles; return C-order [N,C] CPU features."""
    model.to(device=device, dtype=torch.float32).eval()
    shape = prior.shape
    if inference_patch_size is None:
        value = model(torch.from_numpy(prior)[None,None].to(device))
        if value.ndim != 5 or value.shape[0] != 1 or tuple(value.shape[2:]) != shape:
            raise ValueError('Model must preserve the prior spatial shape')
        features = value[0].permute(1,2,3,0).reshape(int(np.prod(shape)),-1).contiguous().cpu()
    else:
        _positive_integer(inference_patch_size, 'inference_patch_size')
        if inference_patch_size < 16 or inference_patch_size % 8:
            raise ValueError('inference_patch_size must be a multiple of 8 and >=16')
        if not isinstance(model, UNet3D):
            raise ValueError('Tiled inference is only supported for UNet3D')
        halo = 64
        features = None
        for start in itertools.product(*(range(0,size,inference_patch_size) for size in shape)):
            stop = tuple(min(s+inference_patch_size,size) for s,size in zip(start,shape))
            lo = tuple(max(0,s-halo) for s in start)
            hi = tuple(min(size,((e+7)//8)*8+halo) for e,size in zip(stop,shape))
            region = tuple(slice(a,b) for a,b in zip(lo,hi))
            value = model(torch.from_numpy(prior[region].copy())[None,None].to(device))[0]
            if features is None:
                features = torch.empty((*shape,value.shape[0]),dtype=torch.float32)
            crop = tuple(slice(s-a,e-a) for s,e,a in zip(start,stop,lo))
            dest = tuple(slice(s,e) for s,e in zip(start,stop))
            features[dest] = value.permute(1,2,3,0)[crop].cpu()
        features = features.reshape(int(np.prod(shape)),-1)
    if not torch.isfinite(features).all().item():
        raise FloatingPointError('Nonfinite deep-kernel features')
    return features


@torch.no_grad()
def construct_deep_kernel(model, prior, *, k=50, local_window=9,
                          include_self=False, chunk_size=512,
                          device=None, inference_patch_size=None):
    """Export a frozen local kernel from the final trained model.

    Args:
        model (torch.nn.Module): Trained feature network, evaluated in eval mode.
        prior (numpy.ndarray): PET prior; min-max normalization matches training.
        k (int): Retained neighbors per voxel; match training settings.
        local_window (int): Odd search width; match training settings.
        include_self (bool): Whether to retain self; match training settings.
        chunk_size (int): Query voxels per kernel export batch.
        device (str | torch.device | None): CPU/CUDA; defaults to model device.
        inference_patch_size (int | None): Optional aligned core tile size
            (multiple of 8, >=16), with a fixed 64-voxel context halo for UNet3D.
            None runs full-volume feature inference. The halo adds memory and
            computation overhead; on small volumes tiles may span the full image.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Float32 weights and int64 C-order
        indices, each [prior.size,k], compatible with ExternalKEMTransform.
    """
    prior = _normalize_image(prior, 'prior')
    _positive_integer(chunk_size, 'chunk_size')
    _validate_neighbors(prior.shape, k, local_window, include_self)
    device = _model_device(model, device)
    features = _infer_deep_features(model, prior, device, inference_patch_size).to(device)
    n = prior.size
    weights = np.empty((n,k),dtype=np.float32)
    indices = np.empty((n,k),dtype=np.int64)
    for start in range(0,n,chunk_size):
        stop = min(n,start+chunk_size)
        rows = torch.arange(start,stop,device=device)
        w, ix = _deep_kernel_rows(features,prior.shape,rows,k=k,
            local_window=local_window,include_self=include_self)
        weights[start:stop] = w.cpu().numpy()
        indices[start:stop] = ix.cpu().numpy()
    return weights,indices
