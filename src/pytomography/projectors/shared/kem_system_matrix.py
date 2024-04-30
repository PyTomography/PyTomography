from __future__ import annotations
from ..system_matrix import SystemMatrix

class KEMSystemMatrix(SystemMatrix):
    """Given a KEM transform :math:`K` and a system matrix :math:`H`, implements the transform :math:`HK` (and backward transform :math:`K^T H^T`)

    Args:
        system_matrix (SystemMatrix): System matrix corresponding to a particular imaging system
        kem_transform (KEMTransform): Transform used to go from coefficient image to real image of predicted counts. 
    """
    def __init__(self, system_matrix, kem_transform):
        self.object_meta = system_matrix.object_meta
        self.proj_meta = system_matrix.proj_meta
        self.system_matrix = system_matrix
        self.kem_transform = kem_transform
        self.kem_transform.configure(system_matrix.object_meta, system_matrix.proj_meta)
        # Inherit required functions from system matrix
        self.set_n_subsets = self.system_matrix.set_n_subsets
        self.get_projection_subset = self.system_matrix.get_projection_subset
        self.get_weighting_subset = self.system_matrix.get_projection_subset
        
    def compute_normalization_factor(self, subset_idx : int | None = None):
        """Function used to get normalization factor :math:`K^T H^T_m 1` corresponding to projection subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Index of subset. If none, then considers all projections. Defaults to None.

        Returns:
            torch.Tensor: normalization factor :math:`K^T H^T_m 1`
        """
        object = self.system_matrix.compute_normalization_factor(subset_idx)
        return self.kem_transform.backward(object)
        
    def forward(self, object, subset_idx=None):
        r"""Forward transform :math:`HK`

        Args:
            object (torch.tensor): Object to be forward projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.

        Returns:
            torch.tensor: Corresponding projections generated from forward projection
        """
        object = self.kem_transform.forward(object)
        return self.system_matrix.forward(object, subset_idx)
    
    def backward(self, proj, subset_idx=None):
        r"""Backward transform :math:`K^T H^T`

        Args:
            proj (torch.tensor): Projection data to be back projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
            return_norm_constant (bool, optional): Additionally returns :math:`K^T H^T 1` if true; defaults to False.

        Returns:
            torch.tensor: Corresponding object generated from back projection.
        """
        object = self.system_matrix.backward(proj, subset_idx)
        return self.kem_transform.backward(object)
        