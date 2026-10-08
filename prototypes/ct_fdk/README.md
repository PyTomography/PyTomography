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
| `low_signal.py` | adaptive filtering of photon-starved rays, from the photons per detector column that DICOM-CT-PD stores (7033,1065) |
| `channel_correction.py` | fit and apply the per-channel scale of the line integrals that matches the scanner (a bowtie-type beam hardening calibration) |

```
cd prototypes/ct_fdk
python run_synthetic.py --rotations 3 --pitch 1
python audit_geometry.py <projections> "as read" "central column -0.5" "z shift sign flipped" --cache scan.pt
python run_vendor_match.py <C145 projections> <C145 full dose images> --cache c145.pt --central-column-offset -0.42
python channel_correction.py <volume on the scanner grid, every slice> <C145 full dose images> --mu-water 0.0186 --exclude=-215,-145 --out c145_fit.json
python run_vendor_match.py <C145 projections> <C145 full dose images> --cache c145.pt --central-column-offset -0.42 --low-signal 30 --channel-correction c145_fit.json
```

## GPU memory

The GPU is shared, so every stage is sized from a budget: 1.5 GB by default, and never more than a quarter of the
memory free when the stage starts (`gpu_budget`). Projections stay on the CPU:

- rebinning copies only the fan views each chunk of parallel angles needs;
- filtering works on chunks of views;
- the back projection's 4D temporaries (views x Nx x Ny x z) are cut into z chunks that fit the budget.

The output volume is the only full-size array on the GPU. Measured peaks on C145, on the scanner grid
(512 x 512 x 40 slices): rebinning 0.32 GB, filtering 0.97 GB, back projection 1.23 GB (`nvidia-smi` shows about
0.7 GB more for the CUDA context and PyTorch's cache). The whole scan (512 x 512 x 316) peaks at 1.27 GB and takes
81 s to back project (305 s with 1.25 mm slices). The synthetic test peaks at 0.65 GB.

A bigger budget does not make it faster. On the 40-slice block, budgets of 1.5, 3 and 6 GB with 4, 16 or 32 views per
batch all took 10.0 to 11.1 s and gave the same image (the 6 GB runs held up to 4.8 GB). The time goes into the
arithmetic of the per-voxel weights (about 40 bytes of temporaries per voxel and view), with the GPU already 96% busy,
so the speed-up has to come from a fused kernel. `torch.compile` would need Triton, which on Windows comes only as the
unofficial `triton-windows` package.

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
| 5. scanner kernel estimated from the images | 54 | -128 | -843 | 1048 | +24 / -16 | 113 | 27 |
| 6. 1.25 mm slices | 54 | -128 | -843 | 1046 | +24 / -16 | 112 | 24 |
| 7. photon-starved rays filtered to about 30 photons | 54 | -128 | -843 | 1046 | +24 / -16 | 112 | 24 |
| 8. **per-channel scale of the line integrals** (fitted outside this block) | **39** | **-110** | **-855** | 1028 | **-6 / +1** | 111 | 24 |

(HU; noise is the SD of the high-pass image in soft tissue. Back projection 10 to 37 s per step.) Over the whole
316-slice scan, the RMS difference goes from 139 HU (step 2) to 108 HU (step 6), 102 HU (step 7) and 101 HU (step 8);
step 7 acts where rays are starved (the shoulders, the arms beside the abdomen), not in this block.

- **Step 1** removes the fan-beam shading: fat moves from -158 to -128 HU, and the edge of the body from -40 to -16 HU.
- **Step 3** is a geometry correction (see the audit below). It cuts the difference from the scanner by 17% and brings
  bone onto the scanner's value.
- **Step 4**: our image was rotated by 0.161 deg relative to the scanner's (0.234 deg before step 3). A rotation of
  the whole geometry leaves the data consistent, so only the comparison with the scanner can find it.
- **Step 5**: the scanner's kernel relative to Ram-Lak, estimated as cross spectrum over power after removing the
  residual 0.44 mm shift (a shift d scales the radially averaged cross spectrum by J0(2 pi f d), 20% at 0.35 cycles/mm;
  it is removed with a phase ramp, whose sign is checked on the data). It is flat to 0.2 cycles/mm, 0.70 at 0.3 and
  zero at 0.4. Above 0.4 cycles/mm the scanner image follows ours with the sign inverted (-0.25 at 0.55 cycles/mm,
  coherence 0.36 to 0.39), which no smooth reconstruction window does. The prototype applies the estimate clipped at
  zero.
- **Step 7** (`low_signal.py`) removes most of the streaks between the shoulders; see below.
- **Step 8** (`channel_correction.py`) removes the radial trend: the soft-tissue difference from the scanner goes from
  +25 HU at the centre and -25 HU at 160 to 180 mm to within +-7 HU (one 20 mm band at -12 HU), and fat, lung and soft
  tissue land within 3 HU of the scanner. See below.

**Still different:**

- A 0.36, 0.25 mm in-plane shift (about half a scanner pixel, 0.33 mm; a half-pixel convention is one candidate).
- The scanner's high-frequency noise texture (43 HU against our 24 HU), and its inverted response above 0.4 cycles/mm.
- Faint z banding: our soft-tissue slice means vary by 2.5 HU from slice to slice (the scanner's by 1.5 HU), most
  strongly at periods near the half-turn table feed (20 mm). The WFBP row weighting is the likely source.

## Photon-starved rays (step 7)

Rays through the shoulders, or through the arms beside the abdomen, expect few photons: DICOM-CT-PD stores the incident
photons per detector column of every view in (7033,1065) PhotonStatistics (54,000 at the centre of the fan and 1,400 at
its edges for C145, with the bowtie and tube current modulation), so the expected count of a ray is N = N0 exp(-p).
At the shoulders 9% of rays expect fewer than 100 photons, and some line integrals reach 17 (no photons at all, clipped).
`low_signal.py` replaces each ray with N below a target by -log of the mean transmission over a neighbourhood
(views x columns x rows) of about target / N rays, blending between box sizes; averaging transmission rather than line
integrals keeps the mean right (on simulated Poisson data at 4 photons: bias +0.12 and SD 0.58 before, no bias and SD
0.17 after, with a target of 30). Rays with enough photons are not touched.

| C145, top 5 cm (shoulders), against the scanner | RMS | RMS after 2 mm blur | soft tissue noise |
|---|---|---|---|
| no filtering | 143 | 49.6 | 57 |
| target 10 photons | 125 | 40.7 | 50 |
| **target 30 photons** | **113** | **36.2** | **39** |
| target 100 photons | 108 | 35.1 | 32 |
| scanner | | | 37 |

A target of 30 photons matches the scanner's noise and brings the shoulders in line with the rest of the scan; 1.2% of
all rays change, in 44 s on the CPU.

## Beam hardening and scatter: a per-channel scale (step 8)

The projections are exported after the scanner's own beam hardening and scatter corrections: the DICOM-CT-PD flags
(7039,1003) BeamHardeningCorrectionFlag and (7039,1008) ScatterCorrectionFlag are YES, as are gain, dark field, flat
field, bad pixel and log. To see what the scanner does beyond that, the smoothed difference (scanner - ours) was
projected along parallel rays through mid-scan slices and fitted against each ray's line integral p, its line integral
through bone, and its distance t from the isocentre (rays that cross the body near the edge of the scanner's image are
left out):

| model of scanner - ours, per ray | R^2 |
|---|---|
| a function of p (water beam hardening) | 0.25 |
| + the line integral through bone (bone beam hardening) | 0.26 |
| + a dependence on t | **0.42** |

Not scatter: that would grow about like exp(p), 20 times from p = 2 to 5, against the 2.8 times observed. Not bone.
At fixed p the difference depends on t: it is a **per-channel scale of the line integrals**, the form a bowtie-dependent
beam hardening calibration takes. Fitted on 20 slices outside the 40-slice block: g(t) = -1.49% + 1.86% (t / 100 mm)^2
(scanner relative to the exported projections; held constant beyond 140 mm, where too few rays are usable), crossing
zero near 90 mm. Applied as p -> p (1 + g(rho sin gamma)) per detector column, it removes the radial trend on the
held-out block (step 8 above).

It matches the scanner; whether the scanner's calibration or the exported one is closer to the truth for this patient
cannot be told without a water phantom scanned on the same GE system. It is fitted on one GE scan, so it needs other GE
cases (with scanner images) before it becomes a default.

## Geometry audit

Every geometric tag of every view was read for both vendors:

| | C145 (GE) | Siemens ACR phantom |
|---|---|---|
| flying focal spot | none | z mode: alternate views at dz = -0.66 mm, drho = +5.45 mm (a 7 deg anode); dphi = 3.35e-4 rad on every view |
| central element (col, row) | 444.75, 32.5 | 369.625, 32.5 |
| views per rotation (tag / angles) | 984 / 984.0 | 2304 / 2304.0 |

Nothing that varies from view to view is ignored. The conventions were then tested by data consistency (OS-SART
2 x 20 on 1 mm voxels unless noted, residual |Hf - g| / |g| x 1000):

| variant | C145 (GE) | C001 (GE) | Siemens |
|---|---|---|---|
| as read | 86.70 | 87.57 | 83.35 |
| no focal spot shifts | | | 83.69 |
| z / radial / angular shift sign flipped | | | 85.15 / 84.00 / 83.51 |
| columns reversed | 213.2 (2 mm voxels; as read 89.7) | | 146.17 |
| rows reversed | 174.9 (2 x 40 subsets; as read 71.7) | | |
| central column -1.0 / -0.75 / -0.5 / -0.25 | 86.76 / 86.20 / **85.98** / 86.09 | - / 87.46 / **87.12** / 87.14 | |
| central column -0.5 / +0.5 | 89.05 / 91.07 (2 mm voxels; as read 89.72) | | 83.11 / 83.08 |
| central column -2.25 (counted from the other end) | | | 91.54 |
| central row +0.25 / +0.5 / +0.75 / +1.0 | 86.59 / 86.53 / 86.50 / 86.71 | | |

- **The flying focal spot is read correctly.** Applying the shifts lowers the residual, and flipping the sign of any
  of them raises it.
- **Column and row directions are right.**
- **GE central column:** both GE scans fit best with the central column about 0.4 channels below the tag (fitted
  -0.46 for C145, -0.39 for C001: 444.3 instead of 444.75), and half a channel above it fits worse. The scanner
  comparison confirms it (step 3 above). Counting the tag from the other end of the detector would give 444.25; for
  the Siemens scan that rule is clearly wrong, and half a channel either way changes its residual by 0.3%, the same
  both ways, so the Siemens tag stands. Until this is confirmed against GE's documentation, both scripts take
  `--central-column-offset`.
- **Central row:** a weak preference for +0.5 to +0.7 rows (0.2%). Not conclusive.
