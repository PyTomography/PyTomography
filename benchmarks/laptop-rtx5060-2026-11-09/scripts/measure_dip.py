"""GPU memory of fitting t_PETGATE_DIP's network, for one START_CHANNELS, in a fresh process.

    python -I measure_dip.py <notebook.ipynb> <outputs dir holding PET/GATE-mMR-BrainInWater/gate_delayeds.pt> <start_channels>

The UNet and DIPPrior classes are taken from the notebook's own cells, the MR goes through the tutorial's pipeline on
the tutorial's grid (204 x 204 x 154, 1.25 mm), and the network is fitted as the tutorial does it twice: pretraining
(L-BFGS, 20 iterations, fixed step) and one refit (10 iterations, strong Wolfe line search). Prints one JSON line.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn  # noqa: F401  (the notebook's cells use it)
from torch.optim import LBFGS  # noqa: F401
import pytomography
from pytomography import io, datasets
from monai.transforms import CropForeground, Compose, DivisiblePadd, SpatialCropd, ThresholdIntensityd, ScaleIntensityd

notebook, outputs, channels = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cells = [''.join(c['source']) for c in json.loads(notebook.read_text(encoding='utf8'))['cells'] if c['cell_type'] == 'code']
ns = dict(torch=torch, nn=torch.nn, LBFGS=LBFGS, pytomography=pytomography)
for src in cells:
    if 'class UNetCustom' in src or 'class DIPPrior' in src:
        exec(src, ns)

_, object_meta, _ = torch.load(outputs / 'PET' / 'GATE-mMR-BrainInWater' / 'gate_delayeds.pt', weights_only=False)
object_meta = object_meta.with_grid(dr=(1.25, 1.25, 1.25), shape=(204, 204, 154))
path = datasets.data_dir() / 'PET' / 'GATE-mMR-BrainInWater'
mri_aligned = io.load_image(path / 'fdg_pet_phantom_mri.nii.gz').on(object_meta)
roi_start, roi_end = CropForeground().compute_bounding_box(mri_aligned.unsqueeze(0))
pipeline = Compose([
    SpatialCropd(['MR', 'NM'], roi_start=roi_start, roi_end=roi_end, allow_missing_keys=True),
    DivisiblePadd(['MR', 'NM'], 16, allow_missing_keys=True),
    ThresholdIntensityd(['MR'], 250, above=False, cval=250),
    ThresholdIntensityd(['MR'], 120, above=True, cval=120),
    ScaleIntensityd(['MR'], 0, 1)])
input_shape = tuple(pipeline({'MR': mri_aligned.unsqueeze(0)})['MR'].shape)

dev = pytomography.device
pet = torch.rand(tuple(object_meta.shape), device=dev) * 0.3   # a stand-in PET image: memory doesn't depend on values
torch.manual_seed(0)
c = channels
net = ns['UNetCustom']([c, 2 * c, 4 * c, 8 * c, 16 * c]).to(dev)
n_params = sum(p.numel() for p in net.parameters())
prior = ns['DIPPrior'](net, mri_aligned, pipeline, n_epochs=1, scale_factor=50, lr=0.01)

out = dict(start_channels=c, input_shape=input_shape, n_params=n_params, gpu=torch.cuda.get_device_name(0),
           baseline_gb=round(torch.cuda.memory_allocated() / 1e9, 3))
try:
    for stage, (n_iter, lr, search) in {'pretrain': (20, 0.01, None), 'refit': (10, 1, 'strong_wolfe')}.items():
        prior.max_iter, prior.lr, prior.line_search_fn = n_iter, lr, search
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        prior.fit(pet)
        torch.cuda.synchronize()
        out[f'{stage}_s'] = round(time.time() - t, 1)
        out[f'{stage}_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out[f'{stage}_reserved_gb'] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
except torch.OutOfMemoryError as e:
    out['error'] = 'OutOfMemoryError: ' + str(e).splitlines()[0][:160]
print(json.dumps(out), flush=True)
