"""Properties every system matrix must have, whatever its geometry.

A new system matrix gets these checks by subclassing the contract in a test module and
implementing ``make()``:

    from contracts import SystemMatrixContract

    class TestPinholeSystemMatrix(SystemMatrixContract):
        def make(self):
            return tiny_pinhole_system()   # (system_matrix, object, projections)

The contract class does not start with "Test", so pytest only collects it through subclasses.
"""
from __future__ import annotations

import torch


class SystemMatrixContract:
    #: Number of subsets used by the subset tests; must divide the number of projection angles
    n_subsets = 3
    #: Relative tolerance of <Hf, g> = <f, H^T g>. Interpolating projectors (bilinear rotation) are adjoint
    #: only to interpolation accuracy, so a subclass may loosen this; say why when you do.
    adjoint_rtol = 1e-4

    def make(self):
        """Return (system_matrix, object, projections) for a small problem."""
        raise NotImplementedError

    def test_forward_and_backward_are_adjoint(self):
        H, f, g = self.make()
        lhs = (H.forward(f) * g).sum()
        rhs = (f * H.backward(g)).sum()
        assert torch.isclose(lhs, rhs, rtol=self.adjoint_rtol), f"<Hf,g> = {lhs.item():.6g}, <f,H^T g> = {rhs.item():.6g}"

    def test_subsets_partition_the_projections(self):
        H, f, g = self.make()
        full_forward = H.forward(f)
        full_backward = H.backward(g)
        H.set_n_subsets(self.n_subsets)
        backward_sum = 0
        for m in range(self.n_subsets):
            torch.testing.assert_close(H.forward(f, m), H.get_projection_subset(full_forward, m), rtol=1e-4, atol=1e-6)
            backward_sum = backward_sum + H.backward(H.get_projection_subset(g, m), m)
        torch.testing.assert_close(backward_sum, full_backward, rtol=1e-4, atol=1e-5)

    def test_non_negative_object_gives_non_negative_projections(self):
        H, f, _ = self.make()
        assert H.forward(f.abs()).min() >= -1e-6
