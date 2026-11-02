"""CGLS on a small SPECT phantom, with an explicitly matched adjoint.

Run ``python examples/cgls_spect.py --output cgls-example`` after installing PyTomography.
No external data or GPU is required. The system matrix is materialised only for this tiny validation problem.
It uses the existing SPECT forward projector and its exact matrix transpose, allowing comparison with NumPy's
SVD least-squares solution. This is not a scalable replacement for the production SPECT backprojector.

The simulated counts are a numerical example, not a clinical acquisition or an ensemble bias study. The script
reports the mismatch of the production backprojector separately. Negative estimates are retained by CGLS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytomography
import torch
from pytomography.algorithms import CGLS
from pytomography.likelihoods import NegativeMSELikelihood
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors import SystemMatrix
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform


class ExplicitSPECT(SystemMatrix):
    """The tiny forward projector materialised column by column, with its exact transpose."""

    def __init__(self, source):
        super().__init__(source.object_meta, source.proj_meta)
        shape = source.object_meta.shape
        columns = []
        for index in range(np.prod(shape)):
            voxel = torch.zeros(shape)
            voxel.reshape(-1)[index] = 1
            columns.append(source.forward(voxel).reshape(-1))
        self.matrix = torch.stack(columns, dim=1)

    def forward(self, object, subset_idx=None):
        return (self.matrix @ object.reshape(-1)).reshape(self.proj_meta.shape)

    def backward(self, projections, subset_idx=None):
        return (self.matrix.T @ projections.reshape(-1)).reshape(self.object_meta.shape)


def run(output: Path):
    """Save reconstruction arrays, independently checked numerical results, and a figure."""
    pytomography.set_device(torch.device("cpu"))
    pytomography.set_dtype(torch.float64)
    torch.set_num_threads(2)
    n, spacing, n_angles = 12, 4.0, 36
    coordinates = (torch.arange(n) - (n - 1) / 2) * spacing
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    body = x.square() + y.square() < 20**2
    truth = (body * 1.0 + ((x + 8).square() + y.square() < 6**2) * 3.0).unsqueeze(-1).repeat(1, 1, 2)
    attenuation = (body * 0.015).unsqueeze(-1).repeat(1, 1, 2)
    object_meta = SPECTObjectMeta((spacing,) * 3, (n, n, 2))
    angles = np.linspace(0, 360, n_angles, endpoint=False) + 0.37
    proj_meta = SPECTProjMeta((n, 2), (spacing, spacing), angles, radii=[150.0] * n_angles)
    source = SPECTSystemMatrix([SPECTAttenuationTransform(attenuation_map=attenuation)], [], object_meta, proj_meta)
    system = ExplicitSPECT(source)
    rng = np.random.default_rng(412)
    f = torch.tensor(rng.normal(size=truth.shape))
    g = torch.tensor(rng.normal(size=proj_meta.shape))
    lhs = torch.sum(source.forward(f) * g)
    production_rhs = torch.sum(f * source.backward(g))
    exact_rhs = torch.sum(f * system.backward(g))
    scale = float(torch.linalg.vector_norm(source.forward(f)) * torch.linalg.vector_norm(g))
    background = torch.full(proj_meta.shape, 0.1)
    expectation = system.forward(truth) + background
    counts = rng.poisson(200 * expectation.numpy())
    data = torch.tensor(counts / 200)
    likelihood = NegativeMSELikelihood(system, data, additive_term=background)
    algorithm = CGLS(likelihood)
    reconstruction = algorithm(1000, rtol=1e-10)
    matrix = system.matrix.numpy()
    reference, _, rank, singular_values = np.linalg.lstsq(matrix, (data - background).numpy().reshape(-1), rcond=None)
    error = np.linalg.norm(reconstruction.numpy().reshape(-1) - reference) / np.linalg.norm(reference)
    results = {
        "image_shape": list(truth.shape),
        "projection_shape": list(data.shape),
        "spacing_mm": spacing,
        "matrix_bytes": system.matrix.numel() * system.matrix.element_size(),
        "rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "production_adjoint_relative_defect": float(abs(lhs - production_rhs)) / scale,
        "exact_adjoint_relative_defect": float(abs(lhs - exact_rhs)) / scale,
        "iterations": algorithm.iterations,
        "converged": algorithm.converged,
        "relative_error_against_lstsq": float(error),
        "normal_residual_norms": algorithm.normal_residual_norms,
        "negative_voxels": int(torch.count_nonzero(reconstruction < 0)),
    }
    assert algorithm.converged
    assert error < 1e-7
    assert results["exact_adjoint_relative_defect"] < 1e-12
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(output / "phantom.npz", truth=truth.numpy(), data=data.numpy(), background=background.numpy(),
                        reconstruction=reconstruction.numpy(), reference=reference.reshape(truth.shape))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, title, values in zip(axes, ["Simulated activity", "CGLS", "CGLS minus SVD solution"],
                                 [truth.numpy(), reconstruction.numpy(), reconstruction.numpy() - reference.reshape(truth.shape)]):
        im = ax.imshow(values[:, :, 0].T, origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Voxel index")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output / "reconstruction.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in results.items() if k != "normal_residual_norms"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cgls-example"))
    run(parser.parse_args().output)
