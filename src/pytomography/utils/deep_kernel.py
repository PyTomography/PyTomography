# Copyright (c) 2026 H. Wei
# SPDX-License-Identifier: MIT
# See the LICENSE file in the repository root for license terms.

"""Patch-trained feature kernels, exported for ExternalKEMTransform.

The default cross-modal objective adapts the user's deep_kem.py:
MSE(K_theta(PET prior) @ SPECT MLEM, PET prior). Images are independently
min-max normalized. 
"""
from __future__ import annotations
import itertools
from numbers import Real
import numpy as np
import torch
from .kernel import _positive_integer
from ..models import UNet3D


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
