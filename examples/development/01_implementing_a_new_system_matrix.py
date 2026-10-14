"""Implementing a new system matrix

Write your own forward and back projector and reuse every algorithm in the library.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_examplesystemmatrix.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Implementing New System Matrices
import pytomography
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.projectors import SystemMatrix
import torch

pytomography.device = 'cpu'

M = 3
object_meta = ObjectMeta(dr=(1.5,1.5,1.5), shape=(M,M,M))

# %% Metadata
class EXSProjMeta(ProjMeta):
    def __init__(self, M, sensitivity_factor):
        self.M = M
        self.sensitivity_factor = sensitivity_factor
        if (sensitivity_factor.shape[0]!=M)*(sensitivity_factor.shape[1]!=M):
            raise ValueError("sensitivity_factor should have side dimensions M")

M = object_meta.shape[0]
# Note: the sensitivty factor is the same for each projection angle, a single detector is "rotating" between angle 0 and 90
sensitivity_factor = torch.ones((M,M))+0.3*torch.rand((M,M))
proj_meta = EXSProjMeta(M, sensitivity_factor)

# %% Part 1: Understanding the Forward/Backward PRojections
sample_object = torch.rand(object_meta.shape) 

# %% Part 2: Implementing The Forward/Backward Projections in the System Matrix Class
class EXSSystemMatrix(SystemMatrix):
    def forward(self, object, subset_idx = None):
        projection_0degrees = object.sum(dim=0)
        projection_90degrees = object.sum(dim=1)
        projections = torch.stack([projection_0degrees, projection_90degrees], dim=0)
        projections *= self.proj_meta.sensitivity_factor
        return projections
    def backward(self, projections, subset_idx = None):
        object_BP_angle0 = (projections[0]*self.proj_meta.sensitivity_factor).unsqueeze(0).repeat(self.proj_meta.M,1,1)
        object_BP_angle90 = (projections[1]*self.proj_meta.sensitivity_factor).unsqueeze(1).repeat(1,self.proj_meta.M,1)
        object_BP = object_BP_angle0 + object_BP_angle90
        return object_BP

system_matrix = EXSSystemMatrix(object_meta=object_meta, proj_meta=proj_meta)
FP = system_matrix.forward(sample_object)
BP = system_matrix.forward(sample_object)

sample_object = torch.rand(object_meta.shape) # object has batch dimension

# %% Part 3: Incorporating Subsets
class EXSSystemMatrix(SystemMatrix):
    def compute_normalization_factor(self):
        norm_projections = torch.ones((2,self.proj_meta.M, self.proj_meta.M))
        return self.backward(norm_projections)
    def forward(self, object, subset_idx = None):
        projection_0degrees = object.sum(dim=0)
        projection_90degrees = object.sum(dim=0)
        if subset_idx==0:
            projections = projection_0degrees
        elif subset_idx==1:
            projections = projection_90degrees
        else:
            projections = torch.stack([projection_0degrees, projection_90degrees], dim=0)
        projections *= self.proj_meta.sensitivity_factor
        return projections
    def backward(self, proj, subset_idx = None):
        # Back projection expects projections in their subset
        if subset_idx is not None:
            object_BP = (proj[0]*self.proj_meta.sensitivity_factor).unsqueeze(subset_idx).repeat_interleave(self.proj_meta.M, subset_idx)
        else:
            object_BP_angle0 = (proj[0]*self.proj_meta.sensitivity_factor).unsqueeze(0).repeat(self.proj_meta.M,1,1)
            object_BP_angle90 = (proj[1]*self.proj_meta.sensitivity_factor).unsqueeze(1).repeat(1,self.proj_meta.M,1)
            object_BP = object_BP_angle0 + object_BP_angle90
        return object_BP

system_matrix = EXSSystemMatrix(object_meta=object_meta, proj_meta=proj_meta)
FP_subset1 = system_matrix.forward(sample_object, subset_idx=1)

# %% Example 2: List Mode System Matrices
detector_ids = torch.randint(low=0, high=2*M**2, size=(400,))

# %% MetaData
scanner_LUT = torch.cartesian_prod(
    torch.tensor([0,1]), # Angle
    torch.arange(M), # row
    torch.arange(M), # column
)

class EXSListmodeProjMeta(ProjMeta):
    def __init__(self, shape, scanner_LUT, detector_ids, sensitivity_factor):
        self.scanner_LUT = scanner_LUT
        self.detector_ids = detector_ids
        self.sensitivity_at_ids = sensitivity_factor[*scanner_LUT[:,1:].T]
        self.shape = shape
        if (sensitivity_factor.shape[0]!=M)*(sensitivity_factor.shape[1]!=M):
            raise ValueError("sensitivity_factor should have side dimensions M")
proj_meta_listmode = EXSListmodeProjMeta((2,M,M), scanner_LUT, detector_ids, sensitivity_factor)

class EXSListmodeSystemMatrix(SystemMatrix):
    # ----
    # SAME AS ABOVE
    # ----
    def forward(self, object, subset_idx = None):
        # There is probably a faster implementation, but I am trying to keep it simple for illustration purposes
        projections = []
        for i, detector_id in enumerate(self.proj_meta.detector_ids):
            coord = self.proj_meta.scanner_LUT[detector_id]
            sensitivity_factor_i = self.proj_meta.sensitivity_at_ids[detector_id]
            if coord[0]==0: # If angle 0:
                projections.append(object[:,coord[1],coord[2]].sum() *  sensitivity_factor_i) # sum along x
            elif coord[0]==1: # If angle 90:
                projections.append(object[coord[1],:,coord[2]].sum() * sensitivity_factor_i)  # sum along y
        return torch.tensor(projections)
    # ---
    # NEW CODE
    # ---
    def backward(self, projections, subset_idx = None):
        # There is probably a faster implementation, but I am trying to keep it simple for illustration purposes
        object = torch.zeros(object_meta.shape)
        projections *= self.proj_meta.sensitivity_at_ids[self.proj_meta.detector_ids]
        for i, detector_id in enumerate(detector_ids):
            coord = self.proj_meta.scanner_LUT[detector_id]
            if coord[0]==0: # If angle 0:
                object[:,coord[1],coord[2]] += projections[i]
            elif coord[0]==1: # If angle 90:
                object[coord[1],:,coord[2]] += projections[i]
        return object

system_matrix = EXSListmodeSystemMatrix(object_meta=object_meta, proj_meta=proj_meta_listmode)
sample_object = torch.rand(object_meta.shape) # object has batch dimension
FP = system_matrix.forward(sample_object)
BP = system_matrix.backward(FP)
