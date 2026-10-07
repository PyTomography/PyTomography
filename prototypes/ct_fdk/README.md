# Helical FBP prototype for third generation CT

A first analytic reconstruction for DICOM-CT-PD data (`CTGen3ProjMeta`: cylindrical detector, helical focal spot
path), and the work of bringing it close to the scanner's own reconstruction. It is **not part of the package**; the
pull request that adds this folder holds the implementation plan.

| File | |
|---|---|
| `fdk_prototype.py` | rebinning to parallel beams, ramp filtering, and voxel-driven weighted back projection (WFBP), all within an explicit GPU budget |
| `run_synthetic.py` | a known phantom forward projected with `CTGen3SystemMatrix`, reconstructed and compared with the truth |
| `run_dicom_ct_pd.py` | reconstruct a DICOM-CT-PD scan on a 1 mm grid |
| `run_vendor_match.py` | the corrections towards the scanner's reconstruction, one at a time, each measured on the scanner's grid |
| `audit_geometry.py` | data-consistency test of how the DICOM-CT-PD geometry is read (flying focal spot, detector centre, column direction) |

```
cd prototypes/ct_fdk
python run_synthetic.py --rotations 3 --pitch 1
python audit_geometry.py <projections> "as read" "central column -0.5" "z shift sign flipped" --cache scan.pt
python run_vendor_match.py <C145 projections> <C145 full dose images> --cache c145.pt --central-column-offset -0.42
```

## GPU memory

The GPU is shared, so every stage is sized from a budget: 1.5 GB by default, and never more than a quarter of the
memory free when the stage starts (`gpu_budget`). Projections stay on the CPU:

- rebinning copies only the fan views each chunk of parallel angles needs;
- filtering works on chunks of views;
- the back projection's 4D temporaries (views x Nx x Ny x z) are cut into z chunks that fit the budget.

The output volume is the only full-size array on the GPU. Measured peaks on C145, on the scanner grid
(512 x 512 x 40 slices): rebinning 0.32 GB, filtering 0.97 GB, back projection 1.23 GB (`nvidia-smi` shows about
0.7 GB more for the CUDA context and PyTorch's cache). The synthetic test peaks at 0.65 GB.

## Method

Fan projections are rebinned to parallel beams by bilinear interpolation in (view, channel), giving parallel angles
theta and a uniform t grid; every parallel ray keeps the focal spot it came from on the helix. The rebinned
projections are weighted by the cone cosine and ramp filtered along t. Each voxel then takes, from every parallel
view, the filtered value where its ray meets the detector, weighted by the WFBP weight: the ray's row weight W(q)
divided by the sum of W(q) over the rays through the voxel at theta + k pi that were measured (Stierstorfer et al.
2004).

The first version of this prototype weighted fan-beam rays this way, after fan-beam filtering. That is not valid: the
filtered fan values of a ray and its conjugate differ, so any split other than a fixed one-half each leaves
low-frequency shading.

## Results

Synthetic phantom (256 x 256 x 64 at 1 mm, noise-free), RMS error of the water background:

| | fan-beam weighting (first version) | WFBP |
|---|---|---|
| circular scan | 1.6 HU | 0.9 HU |
| helical, pitch 1, 32 rows | 29 HU (a +100 HU sphere read +59) | **0.9 HU** (sphere +100.0, inserts exact) |

TCIA LDCT case C145 (GE, chest), 40 slices from z = -199.75 to -160.75 mm, against the scanner's reconstruction on its
own grid. Each step adds one correction to the previous ones. Tissue classes come from the smoothed scanner image; RMS
is over the body.

| step | soft tissue | fat | lung | bone | centre / edge soft tissue vs scanner | RMS vs scanner | noise |
|---|---|---|---|---|---|---|---|
| scanner | 42 | -108 | -853 | 1037 | | | 43 |
| 0. fan-beam weighting, 1 mm grid | 44 | -158 | -853 | 956 | +20 / -40 | 138 | 22 |
| 1. parallel rebinning + WFBP, 1 mm grid | 54 | -128 | -844 | 964 | +24 / -16 | 140 | 23 |
| 2. reconstructed on the scanner grid | 54 | -128 | -844 | 985 | +24 / -16 | 144 | 31 |
| 3. GE central column 444.33 (tag 444.75) | 54 | -128 | -843 | 1023 | +24 / -16 | **119** | 31 |
| 4. focal spot angle offset +0.161 deg | 54 | -128 | -843 | 1025 | +24 / -15 | 116 | 31 |
| 5. scanner kernel estimated from the images | 54 | -127 | -843 | 1039 | +24 / -15 | 113 | 25 |
| 6. 1.25 mm slices | 55 | -127 | -843 | 1037 | +24 / -15 | 112 | 22 |

(HU; noise is the SD of the high-pass image in soft tissue. Back projection 10 to 37 s per step.)

- **Step 1** removes the fan-beam shading: fat moves from -158 to -128 HU, and the edge of the body from -40 to -16 HU.
- **Step 3** is a geometry correction (see the audit below). It cuts the difference from the scanner by 17% and brings
  bone onto the scanner's value.
- **Step 4**: our image was rotated by 0.161 deg relative to the scanner's (0.234 deg before step 3). A rotation of
  the whole geometry leaves the data consistent, so only the comparison with the scanner can find it.
- **Step 5**: the scanner's kernel relative to Ram-Lak, estimated as cross spectrum over power, follows a smooth
  low-pass to 0.35 cycles/mm. The coherence of the two images falls below 0.4 beyond 0.3 cycles/mm, so the scanner's
  fine texture is not a linear filter of ours (vendor noise processing, most likely).

**Still different:**

- A smooth radial trend in soft tissue: +24 HU at the centre, -15 HU at 140 to 180 mm. OS-SART shows the same, with
  the same geometry and any number of iterations, so it lies in the data or the vendor's processing, not in the
  reconstruction.
- A 0.36, 0.25 mm in-plane shift.
- The scanner's high-frequency noise texture.

## Geometry audit

Every geometric tag of every view was read for both vendors:

| | C145 (GE) | Siemens ACR phantom |
|---|---|---|
| flying focal spot | none | z mode: alternate views at dz = -0.66 mm, drho = +5.45 mm (a 7 deg anode); dphi = 3.35e-4 rad on every view |
| central element (col, row) | 444.75, 32.5 | 369.625, 32.5 |
| views per rotation (tag / angles) | 984 / 984.0 | 2304 / 2304.0 |

Nothing that varies from view to view is ignored. The conventions were then tested by data consistency (OS-SART
2 x 20, residual |Hf - g| / |g| x 1000):

| variant | C145 (GE, 1 mm) | C001 (GE, 1 mm) | Siemens (1 mm) |
|---|---|---|---|
| as read | 86.70 | 87.57 | 83.35 |
| columns reversed | 213 (2 mm) | | 146.17 |
| rows reversed | 175 (2 mm, 2 x 40) | | |
| no focal spot shifts | | | 83.69 |
| z / radial / angular shift sign flipped | | | 85.15 / 84.00 / 83.51 |
| central column -0.75 / -0.5 / -0.25 | 86.20 / **85.98** / 86.09 | 87.46 / **87.12** / 87.14 | |
| central column -2.25 (counted from the other end) | | | 91.54 |
| central row +0.25 / +0.5 / +0.75 / +1.0 | 86.59 / 86.53 / 86.50 / 86.71 | | |

- **The flying focal spot is read correctly.** Applying the shifts lowers the residual, and flipping the sign of any
  of them raises it.
- **Column and row directions are right.**
- **GE central column:** both GE scans fit best with the central column about 0.4 channels below the tag (fitted
  -0.46 for C145, -0.39 for C001: 444.3 instead of 444.75). The scanner comparison confirms it (step 3 above). Counting
  the tag from the other end of the detector would give 444.25; for the Siemens scan that rule is clearly wrong.
  Until this is confirmed against GE's documentation, both scripts take `--central-column-offset`.
- **Central row:** a weak preference for +0.5 to +0.7 rows (0.2%). Not conclusive.
