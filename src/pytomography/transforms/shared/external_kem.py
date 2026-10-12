# Copyright (c) 2026 H. Wei
# SPDX-License-Identifier: MIT
# See the LICENSE file in the repository root for license terms.

"""Precomputed sparse kernels for kernel expectation maximization."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np
import torch

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
