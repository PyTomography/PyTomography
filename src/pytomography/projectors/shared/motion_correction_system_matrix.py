from __future__ import annotations
import torch
import pytomography
from pytomography.projectors import  SystemMatrix

class MotionSystemMatrix(SystemMatrix):
    def __init__(self, system_matrices, motion_transforms):
        self.object_meta = system_matrices[0].object_meta
        self.proj_meta = system_matrices[0].proj_meta
        self.system_matrices = system_matrices
        self.motion_transforms = motion_transforms
        for motion_transform, system_matrix in zip(motion_transforms, system_matrices):
            motion_transform.configure(system_matrix.object_meta, system_matrix.proj_meta)
        
    def forward(self, object, angle_subset=None):
        return torch.vstack([H.forward(m.forward(object), angle_subset) for m, H in zip(self.motion_transforms, self.system_matrices)])
    
    def backward(self, proj, angle_subset=None):
        objects = []
        for proj_i, system_matrix, motion_transform in zip(proj, self.system_matrices, self.motion_transforms):
            objects.append(motion_transform.backward(system_matrix.backward(proj_i.unsqueeze(0),angle_subset)))
        return torch.vstack(objects).mean(axis=0).unsqueeze(0)
    
    def get_subset_splits(
        self,
        n_subsets: int
    ) -> list:
        return self.system_matrices[0].get_subset_splits(n_subsets)
    
    def compute_normalization_factor(self, angle_subset: list[int] = None):
        norm_proj = torch.ones((len(self.motion_transforms), *self.proj_meta.shape)).to(pytomography.device)
        return self.backward(norm_proj, angle_subset)