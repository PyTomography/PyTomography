"""Implementing a new system matrix

Write your own forward and back projector and reuse every algorithm in the library.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_examplesystemmatrix.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Implementing New System Matrices
import pytomography
from pytomography.algorithms import OSEM, MLEM
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.projectors import SystemMatrix
from pytomography.likelihoods import NegativeMSELikelihood
import matplotlib.pyplot as plt
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
# Sum object along x to get projection at 0 degrees
sample_projection_0degrees = sample_object.sum(dim=0) * proj_meta.sensitivity_factor
# Sum object along y to get projection at 90 degrees
sample_projection_90degrees = sample_object.sum(dim=1) * proj_meta.sensitivity_factor
# Concatenate to get the full set of projections
sample_projections = torch.stack([sample_projection_0degrees, sample_projection_90degrees], dim=0)
sample_projections.shape

# First adjust projections by sensitivity factor
sample_projections_angle_0_sensitivity_adjusted = sample_projections[0]*proj_meta.sensitivity_factor
# Back project at angle 0 by duplication
sample_object_BP_angle0 = sample_projections_angle_0_sensitivity_adjusted.unsqueeze(0).repeat(object_meta.shape[0],1,1)
# Back project at angle 90 by duplication
sample_projections_angle_90_sensitivity_adjusted = sample_projections[1]*proj_meta.sensitivity_factor
sample_object_BP_angle90 = sample_projections_angle_90_sensitivity_adjusted.unsqueeze(1).repeat(1,object_meta.shape[0],1)
# Back projected object is sum of each
sample_object_BP = sample_object_BP_angle0 + sample_object_BP_angle90

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

class EXSSystemMatrix(SystemMatrix):
    def compute_normalization_factor(self):
        # A clever implementation of this function will only compute the normalization factor once, and then store it for future use (e.g. using a boolean flag)
        norm_projections = torch.ones((2,self.proj_meta.M, self.proj_meta.M))
        return self.backward(norm_projections)
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
norm_factor = system_matrix.compute_normalization_factor()

sample_object = torch.rand(object_meta.shape) # object has batch dimension
sample_projections = system_matrix.forward(sample_object)

# Define system matrix
system_matrix = EXSSystemMatrix(object_meta=object_meta, proj_meta=proj_meta)
# Define likelihood that characterizes measured data (for SPECT/PET, this is PoissonLog, but here we'll use NegativeMSE)
likelihood = NegativeMSELikelihood(system_matrix, projections=sample_projections, scaling_constant=0.01)
# Define 
reconstruction_algorithm = MLEM(likelihood)

recon = reconstruction_algorithm(n_iters=40)

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
FP_subset0 = system_matrix.forward(sample_object, subset_idx=0)
FP_subset1 = system_matrix.forward(sample_object, subset_idx=1)
BP_subset0 = system_matrix.backward(FP_subset0, subset_idx=0)
BP_subset1 = system_matrix.backward(FP_subset0, subset_idx=1)

class EXSSystemMatrix(SystemMatrix):
    # ----
    # NEW METHODS
    # ----
    def set_n_subsets(self, n_subsets):
        self.n_subsets = n_subsets
    def get_projection_subset(self, projections, subset_idx):
        # Called when n_subsets>1 in internal pytomography code, in this case, assumes 2 subsets since thats the only possible number of subsets we have. In general, this should split data evenly (see SPECTSystemMatrix source code)
        return projections[subset_idx].unsqueeze(0)
    def compute_normalization_factor(self, subset_idx = None):
        # This function generally looks the same for all system matrices
        norm_projections = torch.ones((2,self.proj_meta.M, self.proj_meta.M))
        if subset_idx is not None:
            norm_projections = self.get_projection_subset(norm_projections, subset_idx)
        return self.backward(norm_projections, subset_idx)
    def get_weighting_subset(self, subset_idx):
        if subset_idx is None:
            return 1
        elif self.n_subsets==2:
            return 0.5 # equal weighting in this case, in general need to be careful with this
    # ----
    # SAME AS PREVIOUS
    # ----
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

sample_object = torch.rand(object_meta.shape) # object has batch dimension
sample_projections = system_matrix.forward(sample_object)

system_matrix = EXSSystemMatrix(object_meta=object_meta, proj_meta=proj_meta)
likelihood = NegativeMSELikelihood(system_matrix, projections=sample_projections, scaling_constant=0.01)
reconstruction_algorithm = OSEM(likelihood)

recon = reconstruction_algorithm(n_iters=40, n_subsets=2)

# %% Example 2: List Mode System Matrices
detector_ids = torch.randint(low=0, high=2*M**2, size=(400,))
detector_ids

# %% MetaData
scanner_LUT = torch.cartesian_prod(
    torch.tensor([0,1]), # Angle
    torch.arange(M), # row
    torch.arange(M), # column
)
scanner_LUT

scanner_LUT[5]

scanner_LUT[detector_ids[0:5]]

sensitivity_factor[*scanner_LUT[detector_ids][:,:2].T]

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

system_matrix = EXSListmodeSystemMatrix(object_meta=object_meta, proj_meta=proj_meta_listmode)
sample_object = torch.rand(object_meta.shape) # object has batch dimension
sample_projections = system_matrix.forward(sample_object)
sample_projections

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
    # ----
    # NEW CODE
    # ----
    def compute_normalization_factor(self):
        norm_BP = torch.zeros(object_meta.shape)
        # Now we loop through unique detector ids instead
        unique_detector_ids = torch.arange(scanner_LUT.shape[0])
        for i, detector_id in enumerate(unique_detector_ids):
            coord = self.proj_meta.scanner_LUT[detector_id]
            sensitivity_factor_i = self.proj_meta.sensitivity_at_ids[detector_id]
            if coord[0]==0: # If angle 0:
                norm_BP[:,coord[1],coord[2]] += sensitivity_factor_i
            elif coord[0]==1: # If angle 90:
                norm_BP[coord[1],:,coord[2]] += sensitivity_factor_i
        return norm_BP

sensitivity_factor

system_matrix = EXSListmodeSystemMatrix(object_meta=object_meta, proj_meta=proj_meta_listmode)
norm_BP = system_matrix.compute_normalization_factor()

system_matrix = EXSListmodeSystemMatrix(object_meta=object_meta, proj_meta=proj_meta_listmode)
likelihood = NegativeMSELikelihood(system_matrix, scaling_constant=0.01)
reconstruction_algorithm = OSEM(likelihood)

recon = reconstruction_algorithm(n_iters=40)

# %% Part 2: Incorporating Subsets
class EXSListmodeSystemMatrix(SystemMatrix):
    # ----
    # NEW CODE
    # ----
    def set_n_subsets(self, n_subsets):
        self.n_subsets = n_subsets
        idx = torch.arange(proj_meta_listmode.detector_ids.shape[0])
        self.subset_indices_array = torch.tensor_split(idx, self.n_subsets)
    def get_projection_subset(self, projections, subset_idx):
        # Needs to consider cases where projection is simply a 1 element tensor in the numerator, but also cases of scatter where it is a longer tensor
        if (projections.shape[0]>1)*(subset_idx is not None):
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices]
        else:
            proj_subset = projections
        return proj_subset
    def get_weighting_subset(self, subset_idx):
        if subset_idx is None:
            return 1
        else:
            # Fraction of events in the subset
            return self.subset_indices_array[subset_idx].shape[0] / proj_meta_listmode.detector_ids.shape[0]
    def forward(self, object, subset_idx = None):
        # There is probably a faster implementation, but I am trying to keep it simple for illustration purposes
        projections = []
        if subset_idx is None:
            detector_ids_subset = self.proj_meta.detector_ids
        else:
            detector_ids_subset = self.proj_meta.detector_ids[self.subset_indices_array[subset_idx]]
        for i, detector_id in enumerate(detector_ids_subset):
            coord = self.proj_meta.scanner_LUT[detector_id]
            sensitivity_factor_i = self.proj_meta.sensitivity_at_ids[detector_id]
            if coord[0]==0: # If angle 0:
                projections.append(object[:,coord[1],coord[2]].sum() *  sensitivity_factor_i) # sum along x
            elif coord[0]==1: # If angle 90:
                projections.append(object[coord[1],:,coord[2]].sum() * sensitivity_factor_i)  # sum along y
        return torch.tensor(projections)
    def backward(self, projections, subset_idx = None):
        # There is probably a faster implementation, but I am trying to keep it simple for illustration purposes
        object = torch.zeros(object_meta.shape)
        if subset_idx is None:
            detector_ids_subset = self.proj_meta.detector_ids
        else:
            detector_ids_subset = self.proj_meta.detector_ids[self.subset_indices_array[subset_idx]]
        projections *= self.proj_meta.sensitivity_at_ids[detector_ids_subset]
        for i, detector_id in enumerate(detector_ids_subset):
            coord = self.proj_meta.scanner_LUT[detector_id]
            if coord[0]==0: # If angle 0:
                object[:,coord[1],coord[2]] += projections[i]
            elif coord[0]==1: # If angle 90:
                object[coord[1],:,coord[2]] += projections[i]
        return object
    def compute_normalization_factor(self, subset_idx=None):
        fraction_considered = self.get_weighting_subset(subset_idx)
        norm_BP = torch.zeros(object_meta.shape)
        # Now we loop through unique detector ids instead
        unique_detector_ids = torch.arange(scanner_LUT.shape[0])
        for i, detector_id in enumerate(unique_detector_ids):
            coord = self.proj_meta.scanner_LUT[detector_id]
            sensitivity_factor_i = self.proj_meta.sensitivity_at_ids[detector_id]
            if coord[0]==0: # If angle 0:
                norm_BP[:,coord[1],coord[2]] += sensitivity_factor_i
            elif coord[0]==1: # If angle 90:
                norm_BP[coord[1],:,coord[2]] += sensitivity_factor_i
        return norm_BP * fraction_considered

system_matrix = EXSListmodeSystemMatrix(object_meta=object_meta, proj_meta=proj_meta_listmode)
likelihood = NegativeMSELikelihood(system_matrix, scaling_constant=0.01)
reconstruction_algorithm = OSEM(likelihood)

recon_2subsets = reconstruction_algorithm(n_iters=20, n_subsets=2)
reconstruction_algorithm = OSEM(likelihood)
recon_1subset = reconstruction_algorithm(n_iters=40)
