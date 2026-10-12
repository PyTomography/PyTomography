# Copyright (c) 2026 H. Wei
# SPDX-License-Identifier: MIT
# See the LICENSE file in the repository root for license terms.

"""Construct kernel from prior image independently of reconstruction."""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np
import torch
from scipy.spatial import cKDTree


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
