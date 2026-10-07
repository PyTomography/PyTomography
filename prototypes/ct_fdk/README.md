# Helical FDK prototype for third generation CT

A first analytic reconstruction for DICOM-CT-PD data (`CTGen3ProjMeta`: cylindrical detector, helical focal spot path),
written to find out how much work a proper implementation is. It is **not part of the package**; the pull request that
adds this folder holds the implementation plan.

| File | |
|---|---|
| `fdk_prototype.py` | equiangular ramp filtering, voxel-driven back projection with `torch.nn.functional.grid_sample`, helical redundancy weights normalized over equivalent rays |
| `run_synthetic.py` | a known phantom forward projected with `CTGen3SystemMatrix` on a GE-like geometry, reconstructed and compared with the truth |
| `run_dicom_ct_pd.py` | a DICOM-CT-PD scan, optionally compared with the scanner's own reconstruction through `get_patient_affine` |

```
cd prototypes/ct_fdk
python run_synthetic.py --rotations 1 --pitch 0
python run_synthetic.py --rotations 3 --pitch 1
python run_dicom_ct_pd.py <C145 projections> --images <C145 full dose images> --cache c145.pt
```

## Method

The filtered projections are those of the equiangular fan beam formula (Kak & Slaney ch. 3): pre-weight by
D cos(gamma) and the cone cosine, convolve each detector row with g(n a) = 0.5 (n a / sin n a)^2 h(n a), apodize
(Hann by default). Each voxel then takes, from every view whose cone reaches it, the filtered value where the ray
from the focal spot through it meets the detector, weighted by 1/L^2.

For the helix, each view's contribution is also weighted by its row window divided by the sum of the row windows of
all rays that measure the same in-plane line through the voxel: views beta + 2 pi m in the same direction and
beta + pi + 2 gamma + 2 pi m in the opposite one. Those are exact because in-plane the focal spot travels a circle; their
heights come from the helix. Each view touches only the slab of slices its cone can reach.

## Results (RTX 5090)

Synthetic phantom, 256 x 256 x 64 at 1 mm, noise-free (`run_synthetic.py`):

| | circular, central slice | helical, pitch 1 (3 rotations, 32 rows) |
|---|---|---|
| water background (truth 0 HU) | +1.0 HU, SD 1.3 | -13.4 HU, SD 26 (RMS error 29 HU over 40 slices) |
| +1000 HU insert | +1002 | +1002 |
| -500 HU insert | -499.5 | -495.1 |
| +100 HU, 20 mm sphere | +101 | +59 (it sits on -40 HU of shading) |
| time | 0.1 s filter, 1.2 s back projection | 0.1 s, 2.9 s |

TCIA LDCT case C145 (GE, chest, 9,014 views, 512 x 512 x 384 at 1 mm), against the scanner's own reconstruction, tissue
classes from the scanner image:

| | scanner | FDK prototype | OS-SART 3 x 40 | OS-SART 10 x 40 |
|---|---|---|---|---|
| soft tissue | 43 HU | 41 HU | 55 HU | 54 HU |
| fat | -109 HU | -146 HU | -119 HU | -122 HU |
| lung | -849 HU | -851 HU | -841 HU | -847 HU |
| in-plane blur against the scanner image | | 1.2 mm | 2.2 mm | 1.3 mm |
| time | | 0.9 s filter + 133 s back projection | 32 s | ~90 s |

**What works:** filtering, geometry and scale are right (the circular case is exact to 1 to 2 HU), and on real data
the image is sharp and soft tissue and lung match the scanner better than OS-SART does.

**What does not, yet:** the helical redundancy weights depend on the voxel, so they are applied after filtering, and
that leaves low-frequency shading of tens of HU (fat, mostly peripheral, comes out 37 HU low on C145; wider row windows
made it worse). The back projection is a Python loop over batches of 4 views (59 ms each); a fused GPU kernel should
take seconds. No flying focal spot (Siemens), no short-scan weighting, one kernel shape.
