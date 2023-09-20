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
    def forward(self, object, angle_subset=None):
        """Forward transform :math:`HK`

        Args:
            object (torch.tensor): Object to be forward projected
            angle_subset (Sequence, optional): Angles to forward projected; if none, project to all angles. Defaults to None.

        Returns:
            torch.tensor: Corresponding projections generated from forward projection
        """
        object = self.kem_transform.forward(object)
        return self.system_matrix.forward(object, angle_subset)
    def backward(self, proj, angle_subset=None, return_norm_constant = False):
        """Backward transform :math:`K^T H^T`

        Args:
            proj (torch.tensor): Projection data to be back projected
            angle_subset (Sequence, optional): Angles corresponding to projections; if none, then all projections from ``self.proj_meta`` are contained. Defaults to None.
            return_norm_constant (bool, optional): Additionally returns :math:`K^T H^T 1` if true; defaults to False.

        Returns:
            torch.tensor: Corresponding object generated from back projection.
        """
        if return_norm_constant:
            object, norm_constant = self.system_matrix.backward(proj, angle_subset, return_norm_constant)
            object, norm_constant = self.kem_transform.backward(object, norm_constant)
            return object, norm_constant
        else:
            object = self.system_matrix.backward(proj, angle_subset, return_norm_constant)
            return self.kem_transform.backward(object)
        