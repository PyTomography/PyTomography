from __future__ import annotations
import torch
from .callback import Callback

class DataStorageCallback(Callback):
    def __init__(self, likelihood, object_initial):
        self.object_previous = torch.clone(object_initial)
        self.objects = []
        self.projections_predicted = []
        self.likelihood = likelihood

    def run(self, object, n_iter, n_subset):
        # Append from previous iteration
        self.objects.append(self.object_previous.cpu())
        # FP contains scatter
        self.projections_predicted.append(self.likelihood.projections_predicted.cpu())
        self.object_previous = torch.clone(object)
        return object
        
    def finalize(self, object):
        self.objects.append(object.cpu())
        self.projections_predicted.append(None)