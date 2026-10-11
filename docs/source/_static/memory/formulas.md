# Memory formulas

These are PyTomography's memory estimates (`estimate_memory()`) and plans (`plan_memory()`), written out.
The page's calculator evaluates the same formulas from `model.json`.
`docs/tools/check_memory_model.py` checks them against the library: every estimate within 1%, every plan the same.

## How an estimate is built

An estimate is a list of parts. Each part is an array, or a group of arrays.
Each part has a scope:

- **held**: kept for the whole run (the image, the data, what the transforms keep);
- **subset**: one subset's arrays, held while that subset is processed;
- **chunk**: the temporaries of one projector call, one view, or one chunk of work;
- **overhead**: Python, PyTorch and CUDA, and what the allocators keep.

The peak is the sum of all the parts, whatever their scope.
Each part is in RAM or on the GPU.
"On the device" means on the GPU when PyTomography's device is CUDA, and in RAM when it is the CPU.
An array you pass in counts at its own size: a view (e.g. one energy window of a stacked tensor) counts as itself, not as the tensor it is taken from.

Three overheads are added to every estimate:

- **Python, PyTorch and CUDA**: $F = 3.7$ GB with a CUDA device, $2.2$ GB without (measured on Windows).
- **PyTorch's GPU cache**: $0.05$ GB plus 20% of the GPU arrays, if there are any.
- **Memory the allocators keep (Windows only)**: a fraction $a$ of all the arrays, RAM and GPU.
  $a = 0.35$, and $a = 0.6$ for PET (sinogram, list mode, scatter), measured on the PET tutorials.

On Windows, GPU memory that PyTorch holds also counts as the process's RAM. So the RAM there includes the GPU.

```math
\mathrm{GPU} = G + [G > 0]\,(0.05\,\mathrm{GB} + 0.2\,G), \qquad G = \sum_{\text{parts}} g_p
```

```math
\mathrm{RAM} = \sum_{\text{parts}} r_p + F + [\text{Windows}]\left(a \sum_{\text{parts}} (r_p + g_p) + \mathrm{GPU}\right)
```

Some steps split their work by a GPU budget.
It is $B_\mathrm{GPU} = \min(1.5\,\mathrm{GB},\ \tfrac14 \times \text{free GPU memory})$ on a CUDA device, and 1.5 GB otherwise.

Notation used below:

- $I = 4 L_x L_y L_z$: one image (float32).
- $I_\mathrm{pad} = 4 (L_x + 2)(L_y + 2)(L_z + 2)$: the image with one voxel of zeros on every face (PET and CT projectors).
- $n$: subsets (`n_subsets`). $s$: projector splits (`N_splits`).
- All sizes are in bytes. $\lceil\cdot\rceil$ and $\lfloor\cdot\rfloor$ round up and down.

## How a plan is chosen

A plan searches the settings of a model in a fixed order. The first setting is the outer loop.
Each setting's candidates are listed fastest first.
The plan estimates each combination in turn, and takes the first that fits both budgets.
It fits when the RAM is at most the RAM budget and the GPU memory at most the GPU budget.
On Windows the RAM budget covers the GPU memory too, because the estimate's RAM includes it.

If nothing fits, the plan takes the combination that exceeds the budgets least:

```math
\text{excess} = \max(0, \mathrm{RAM} - \text{RAM budget}) + \max(0, \mathrm{GPU} - \text{GPU budget})
```

Settings given as fixed keep their value. MLEM always has one subset.

The usual order, for reconstructions with projector splits:

1. `N_splits`: 1, 2, 4, …, 256 and the present value. Fewer splits are faster.
2. `n_subsets`: every number from 1 to the number of views (at most 256). Fewer subsets are faster per iteration.

Every number of subsets is tried, in order. Memory need not fall with more subsets:
CT and SPECT keep one $H_m^T 1$ image per subset.

## Reconstruction algorithms (all iterative models)

What the algorithm adds to its system matrix's parts:

| Part | Scope | Bytes | Where |
|---|---|---|---|
| projections (the data) | held | the data's size | where the data is |
| additive term | held | its size | where the data is |
| prior: its copy of the image, field of view scale and gradient | held | $3I$ | device |
| prior: its neighbour weights (anatomical) | held | $I$ on the device, $27 L_x L_y L_z$ in RAM | |
| prior: temporaries of its gradient | subset | $6I$ | device |
| $H_m^T 1$ of the $n$ subsets stacked, and their sum (BSREM, RBI-EM, RBI-MAP) | subset | $(n + 1) I$ when $n > 1$ | device |

A `DataStorageCallback` is not counted. It keeps an image and the expected projections of every subiteration.

## PET sinogram (PETSinogramSystemMatrix)

The sinogram of $C$ crystals per ring and $R$ rings, as PyTomography's metadata computes it:

```math
N_\theta = \lfloor C/2 \rfloor, \quad N_r = C + 1, \quad P = R^2, \quad S = 4 N_\theta N_r P
```

$S$ is one non-TOF sinogram. With TOF it has $T$ bins; without, $T = 1$.

One subset has $\ell = \lceil N_\theta / n \rceil\, N_r P$ lines of response.
The projector handles $b = 88 + 8T$ bytes a line, in $k = \max\left(s, \lceil \ell b / B_\mathrm{GPU} \rceil\right)$ chunks.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| sensitivity sinogram | held | $S$ | the projections' device |
| normalisation images ($n$) | held | $nI$ | device |
| image, update and attenuation map | held | $4I$ | device |
| data, expected counts, ratio and temporaries (one subset) | subset | $12\,\ell T$ | the projections' device |
| projector chunk | chunk | $\lceil \ell / k \rceil\, b$ | device |
| projections (the data) | held | dense: $S T$; lazy: $(4 + 2[\mathrm{TOF}])\,E + 8(N_\theta + 1)$ | RAM, or GPU if dense there |
| additive term | held | dense: $S T$; lazy: $2S + 4 N_\theta N_r R_s^2 T$ | where the data is; RAM if lazy |

The projections' device is the system matrix's `device` (the GPU, or `'cpu'`).
A lazy sinogram binned from $E$ events keeps 4 bytes an event, 6 with TOF.
A lazy additive term is randoms plus a lazy scatter estimate, divided by the sensitivity, as the TOF tutorials build it.
It keeps two non-TOF sinograms and the scatter estimate's interpolation over $R_s$ sampled rings
($R_s = \lceil R/\sigma \rceil + [(R - 1) \bmod \sigma \neq 0]$ for a ring step $\sigma$).

Windows allocator fraction: 0.6. Settings: `N_splits`, then `n_subsets` from 1 to $\min(256, N_\theta)$.

## PET list mode (PETLMSystemMatrix)

$E$ events, each with $q$ detector IDs of $w$ bytes ($q = 2$, or 3 with TOF; $w = 4$ for int32, 8 for int64).
A subset has $e = \lceil E/n \rceil$ events.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| events and their projector order | held | $E q w + 4E\,[\text{sort\_events}]$ | where the events are (`lor_device`) |
| sensitivity weights | held | $4W$; $W$ = every crystal pair, $CR(CR-1)/2$, unless given | RAM or GPU |
| detector IDs of the sensitivity weights (`detector_ids_sensitivity`, e.g. GE HDF5) | held | $2W$ IDs: $8W$ (int32) or $16W$ (int64) | RAM |
| sensitivity and normalisation images ($n$) | held | $(n+1) I$ | device |
| image, update and attenuation map | held | $4I$ | device |
| expected counts, ratio and temporaries (one subset) | subset | $16e$ | the projections' device |
| positions of the subset's events | subset | $8e$ | where the events are |
| building a subset's projector order (first projection) | subset | $q w e + 130 \min(e, 2^{22}) + 40 \min(e, 2^{24})$ | where the events are; with `sort_events` |
| projector chunk | chunk | $64 \lceil e/s \rceil$ | device |
| projections (the data) | held | 4 (a single value) | device |
| additive term | held | $4E$ | device |

Windows allocator fraction: 0.6. Settings: `N_splits`, then `n_subsets` from 1 to 256.

## Single scatter simulation (get_sss_scatter_estimate)

The scanner is sampled every $\sigma_z$ rings and every $\sigma_c$ crystals:

```math
R_s = \lceil R/\sigma_z \rceil + [(R-1) \bmod \sigma_z \neq 0], \quad C_s = \lceil C / \sigma_c \rceil, \quad
L = \tfrac12 R_s C_s (R_s C_s - 1), \quad Q = \tfrac12 C_s (C_s - 1)
```

$L$ is the sampled lines of response, $Q$ the sampled (angle, radius) positions.
The dense estimate is $D = 4 N_\theta N_r P T$ bytes. Its interpolation over angle and radius is $\rho = 4 N_\theta N_r R_s^2 T$.
`lazy` unset: lazy when $D$ is more than a quarter of the memory budget, or more than 8 GB without one.
`N_splits` is clamped to $1 \ldots T$.

Held:

| Part | Bytes | Where |
|---|---|---|
| PET image and attenuation map | $2I$ | device |
| data (sinogram input) | dense: $D$; lazy: $(4 + 2[\mathrm{TOF}])E + 8(N_\theta+1)$ | RAM, or GPU if dense there |
| randoms | $S$ | RAM |
| events binned lazily (list mode) | $8E + 8(N_\theta + 1)$ | RAM |
| sensitivity sinogram | $S$ | RAM (list mode); the projections' device (sinogram) |
| interpolated estimate | lazy: $\rho$; dense: $2D$ (and its scaled copy) | RAM |

Then four steps, each with host bytes $h$ and device bytes $d$:

- **sparse estimate**: $h = 112 L$, $d = L\,(129 + 8T + [\mathrm{TOF}](28T + [\text{no fused kernel}](4 \lceil T/s \rceil N_d + 16 N_d)))$.
  $N_d$ is `num_dense_tof_bins`. The fused CUDA kernel runs with TOF on a GPU with CuPy, for $T \le 64$.
- **interpolation over angle and radius**: $h = \rho + 32\,C(C-1)/2$,
  $d = 3 \cdot 4 Q R_s^2 T + \max\left(3.5 \cdot 4 (Q+3)^2,\ 12 c (Q + 3) + 8 c R_s^2 T\right)$ with $c = \min(8192, C(C-1)/2)$.
- **interpolation over the ring pairs** (dense only): $h = \rho$, $d = a \cdot 12 N_r P T$ for
  $a = \min(N_\theta, \max(1, \lfloor B_\mathrm{GPU} / (12 N_r P T) \rfloor))$ angles at once.
- **scaling** in $m$ subsets: $h = 32E$ (list mode), $d = 12 \ell T + 24 \ell + \lceil \ell/k \rceil (88 + 8T) + 6I$,
  with $\ell = \lceil N_\theta / m \rceil N_r P$ and $k = \max(1, \lceil \ell (88+8T) / B_\mathrm{GPU} \rceil)$.
  $m = 20$ without a memory budget, else $\max(20, \lceil 2D / (B/2) \rceil)$; on a GPU at least $\lceil D / (B_\mathrm{GPU}/3) \rceil$; at most $N_\theta$.

On the host, only the largest step counts (scope chunk). On the GPU, every step counts:
PyTorch's cache keeps each step's blocks until the estimate ends. Without a GPU, $h + d$ of the largest step is in RAM.

Windows allocator fraction: 0.6.
Settings: `lazy` (dense first: it is read faster afterwards), then `N_splits` from 1 to $T$ (TOF only).

## Helical CT OS-SART (CTGen3SystemMatrix)

$V$ views of $D = $ columns $\times$ rows rays. A subset has $v = \lceil V/n \rceil$ views.
A projector call handles at most $2^{25}$ rays: $c(x) = \max\left(s, \lceil x D / 2^{25} \rceil\right)$ calls for $x$ views.
The largest call has $\rho = \max\left(\lceil v / c(v) \rceil D,\ \lceil V / c(V) \rceil D\right)$ rays (the initial image projects every view).

| Part | Scope | Bytes | Where |
|---|---|---|---|
| image | held | $I$ | device |
| $H_m^T 1$ of $n$ subsets | held | $nI$ | RAM (kept on the host) |
| projections of a subset: measured, predicted, of ones, ratio | subset | $16 v D$ | the projections' device |
| images of an update | subset | $3I$ | device |
| projector call: rays and the padded image | chunk | $28\rho + I_\mathrm{pad} + I$ | device |
| projections (the data) | held | $4VD$ | where they are |

Settings: `N_splits`, then `n_subsets` from 1 to $\min(256, V)$.

## Helical CT filtered back projection (WFBP)

The fan views are rebinned to parallel views, filtered and back projected, a chunk at a time.
Geometry (one focal spot): $\Delta\varphi = $ column spacing / DSD, $c_0$ the central column, $\beta = 2\pi / $ views per rotation.

```math
\gamma = \min(|1 - c_0|, |N_\mathrm{col} - c_0|)\,\Delta\varphi, \quad
\Delta\theta = \pi / \mathrm{round}(\pi/\beta), \quad
J = \left\lfloor \frac{(V - 1)\beta - 2\gamma}{\Delta\theta} \right\rfloor + 1, \quad
M = 2 \left\lceil \frac{\sin\gamma}{\Delta\varphi} \right\rceil
```

$J$ parallel views of $M$ bins and $N_\mathrm{row}$ rows; a view is $u = 4 M N_\mathrm{row}$ bytes.
The back projection looks $k$ half turns either side:
$k = \max(1, \min(\lceil r / (f/2) \rceil, \lceil (J-1)\Delta\theta/\pi \rceil))$,
with $r = (N_\mathrm{row}/2 + 1)\,\Delta_\mathrm{row} (\rho_f + r_\mathrm{FOV}) / \mathrm{DSD}$ + half the slice thickness, and $f$ the table feed per rotation.

A chunk has $j = \min(J, \max(1, \lfloor A / u \rfloor))$ views, $A = B/8$ with a memory budget $B$, else 1 GB.
Its stages, for the GPU budget $g$ ($\min(g, \tfrac14$ free$)$ on a GPU):

- rebinning: $c_1 = \min(j, \max(1, \min(256, \lfloor g / (4 N_\mathrm{col} N_\mathrm{row} + 6u) \rfloor - w)))$, $w = \lceil 2\gamma/\beta \rceil + 3$;
  device $(c_1 + w + 3)\,4 N_\mathrm{col} N_\mathrm{row} + 6 c_1 u$, host $(c_1 + w + 3)\,4 N_\mathrm{col} N_\mathrm{row} + c_1 u$;
- filtering: $p = 2^{\lceil \log_2 2M \rceil}$, $f_v = 2u + 24 p N_\mathrm{row}$, $c_2 = \min(j, \max(1, \min(1024, \lfloor g / f_v \rfloor)))$; device $c_2 f_v$, host $c_2 u$;
- back projection: with the fused CUDA kernel (CuPy, $k \le 8$) $I + \min(J_b, j)\,u$, $J_b = \max(1, \min(128, \lfloor (g - 2I)/(2u) \rfloor))$;
  else $96 L_x L_y (2k+1) + 4 e L_x L_y \min(z, L_z)$ with $e = 40$ on a GPU, 60 on the CPU,
  $z = \max(1, \lfloor (g - I - 96 L_x L_y (2k+1)) / (160 L_x L_y) \rfloor)$.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| projections | held | $4 V N_\mathrm{col} N_\mathrm{row}$ | where they are |
| projections: float32 copy on the host | held | the same, if they are on the GPU | RAM |
| image | held | $I$ | device |
| rebinned views of one chunk | chunk | $j u$ | RAM |
| views being rebinned or filtered (GPU runs) | chunk | rebinning host + filtering host | RAM |
| rebinning, filtering and back projection buffers | chunk | GPU: rebinning + max(filtering, back projection); CPU: their sum if more than one chunk, else the largest | device |

Settings, in order: `gpu_budget` (the given one, 1.5 GB by default, then halves down to a 64th) is the outer loop;
`memory_budget_gb` (the present one, then budgets that halve the chunk, down to a 256th) the inner.
So the memory budget comes down first. Smaller chunks cost little time; a smaller GPU budget costs much more.
Neither changes the image.

Not modelled: a flying focal spot. Each focal spot group then has its own geometry; the model uses one group.

## Cone-beam CT OS-SART (CTConeBeamFlatPanelSystemMatrix)

$D = $ columns $\times$ rows rays a view, $v = \lceil V/n \rceil$.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| image | held | $I$ | device |
| $H_m^T 1$ of $n$ subsets | held | $nI$ | device |
| projections of a subset: measured, predicted, of ones, ratio | subset | $16 v D$ | the projections' device |
| images of an update | subset | $3I$ | device |
| one view: rays and the padded image | chunk | $28 D + I_\mathrm{pad} + I$ | device |
| projections (the data) | held | $4VD$ | where they are |

Settings: `N_splits` (no change in memory), then `n_subsets` from 1 to $\min(256, V)$.

## Cone-beam CT filtered back projection (FDK)

| Part | Scope | Bytes | Where |
|---|---|---|---|
| projections | held | $4VD$ | where they are |
| image | held | $I_\mathrm{pad} + I$ | device |
| distance weights of every view | held | $4 V L_x L_y$ | device |
| one view: back projection, rays, distance weight | chunk | $\max(I_\mathrm{pad} + 28D + 16 \cdot 2^{\lceil \log_2 2N_\mathrm{col} \rceil} N_\mathrm{row} + 3I,\ 8 V L_x L_y)$ | device |

No settings: the plan says whether it fits.

## SPECT (SPECTSystemMatrix)

The object is padded so that rotating it loses nothing: $p = \lceil (\sqrt2 L_x - L_x)/2 \rceil$,
$X = L_x + 2p$, $Y = L_y + 2p$, $O = 4 X Y L_z$ bytes.
The projections have $L_r \times L_{z'}$ bins: $q = 4 L_r L_{z'}$ a view, $q_p = 4 (L_r + 2p_r) L_{z'}$ padded
($p_r = \lceil (\sqrt2 L_r - L_r)/2 \rceil$). $V$ views, $v = \lceil V/n \rceil$ in a subset.

What the object transforms keep ($K$) and the most one call holds ($t$):

- attenuation: its map $I$, plus $V O$ with `cache_probabilities`; $t = O$ cached, $6O$ otherwise;
- Gaussian PSF: $8 X \bar k\, n_r$ ($n_r$ distinct radii, $\bar k$ the mean 1D kernel size); $t = 3O$;
- PSF operator (kernel $k = L_x - 1$, $h = Y + k - 1$, $w = L_z + k - 1$ rounded up to even):
  $n_r(4X + 8k)$, plus $8 X h (w/2 + 1)\, n_r$ with `cache_kernel`; $t = 32 X h w$.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| image | held | $I$ | device |
| $H_m^T 1$ of $n$ subsets | held | $nI$ | device |
| images of an update | held | $3I$ | device |
| rotation grids of every view (cached) | held | $8 V X Y (2 + [\text{attenuation}] + [\text{camera path}])$ | device |
| what the object transforms keep | held | $K$ | device |
| projections of a subset: measured, expected, additive term, of ones, padded copy | subset | $\max(5 v q_p,\ V q + 2 v q_p)$ | device |
| padded object, support and sum of a projection | subset | $3O$ | device |
| one view: rotated object and the transforms' temporaries | chunk | $O + \max(t, O)$ | device |
| projections (the data), additive term | held | $Vq$ each | device |

Settings: `n_subsets` from 1 to $\min(256, V)$.

## SPECT filtered back projection

| Part | Scope | Bytes | Where |
|---|---|---|---|
| projections | held | $Vq$ | device |
| image and its support | held | $2O$ | device |
| rotation grids of every view | held | $8 V X Y$ | device |
| one view: spread and rotated | chunk | $6O$ | device |

No settings.

## StarGuide (StarGuideSystemMatrix)

$A$ angles, at most $B$ views at one angle, $V$ views in all. Subsets are made of whole angles:
a subset has at most $v = \min(V, \lceil A/n \rceil B)$ views. A view is $q = 4 \cdot 16 L_z$ bytes.
Each view samples a slab $W = 18 + 2\lfloor k/2 \rfloor$ columns wide ($k$ the largest PSF kernel; $W = 18$ without PSF).
A slab of an angle is $\Sigma = 4 B L_x W L_z$ bytes.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| image, $H_m^T 1$ of $n$ subsets, images of an update | held | $I$, $nI$, $3I$ | device |
| geometry and PSF kernels of every view (cached) | held | $V(12W + 128 L_x + 8 L_x W + 8 + 8 L_x k)$ | device |
| rotation grids of every angle (cached) | held | $8 A L_x L_y (2 + [\text{attenuation}])$ | device |
| what the transforms keep | held | $I (1 + [\text{cached}] A)$ + $8 L_x \bar k\, n_r$ | device |
| projections of a subset: measured, expected, additive term, scaled, of ones | subset | $\max(5vq,\ Vq + 2vq)$ | device |
| images of a projection: the result and an angle | subset | $2I$ | device |
| one angle: rotated object, slabs, sampled columns and the transforms' temporaries | chunk | $3I + 3\Sigma + 5\Sigma[\text{PSF}] + 128 B L_z L_x + t_a$ | device |
| projections (the data), additive term | held | $Vq$ each | device |

$t_a = 6I$ for attenuation ($I$ with its cache). $\bar k$ is the mean kernel size over the radii ($k$ if not given).
Settings: `n_subsets` from 1 to $\min(256, A)$.

## Kernel EM (KEMSystemMatrix around SPECT)

The SPECT system matrix's parts, then:

| Part | Scope | Bytes | Where |
|---|---|---|---|
| KEM kernel | held | $\text{size}^3\, I$ | RAM, or device with `kernel_on_gpu` |
| KEM support images | held | (number of them) $\times I$ | where they are |
| applying the KEM kernel: result, neighbour, kernel slice, product | chunk | $4I$ | device |

Then the data and the additive term. Settings as SPECT.
Computing the kernel itself (on the host, before the reconstruction) is not estimated.

## Multi-bed SPECT (PGAAMultiBedSPECT)

Every bed's reconstruction stays alive; one bed at a time is updated.

| Part | Scope | Bytes | Where |
|---|---|---|---|
| bed $i$: what its reconstruction holds (one part per bed) | held | the SPECT held parts, its data, additive term and prior | device |
| the SPECT subset and chunk parts, and the algorithm's (of the bed that needs most) | subset, chunk | as SPECT | device |
| the beds' images, stacked, and the stitched image | subset | $3 \times \text{beds} \times I$ | device |

Settings: `n_subsets`, as SPECT.

## Dual-peak or motion SPECT (ExtendedSystemMatrix, MotionSystemMatrix)

$N$ system matrices (one per photopeak, or per gate). The first one's parts, with its subset's projections
counted $2N$ times (each one's projections are kept, then stacked). Then:

| Part | Scope | Bytes | Where |
|---|---|---|---|
| system matrix $i$: what it holds ($i = 2 \ldots N$, distinct ones) | held | $K$ | device |
| arrays of the $N$ transforms (motion) | held | $6 N I$ (a forward and a backward field) | device |
| a transform's temporaries (motion) | chunk | $12 I$ | device |
| projections of ones of all $N$ (for $H^T 1$) | subset | $4 N V L_r L_{z'}$ | device |
| projections (the data), additive term | held | $N V q$ each | device |

## SPECT with SIMIND Monte Carlo scatter (MonteCarloHybridSPECTSystemMatrix)

The SPECT parts (it always starts from the camera path), then, for $\omega$ energy windows and $\pi$ SIMIND processes:

| Part | Scope | Bytes | Where |
|---|---|---|---|
| attenuation map at 140 keV | held | $I$ | device |
| SIMIND maps and the projections of every window read back (host) | subset | $2I + 4\omega v q$ | RAM |
| SIMIND processes ($\pi$) | chunk | $\pi (0.05\,\mathrm{GB} + 2I + 3\omega v q)$ | RAM (of other processes) |

## What the formulas leave out

- Whatever else the script keeps: other images, earlier reconstructions, data not passed to the reconstruction,
  the larger tensor a view you pass in is taken from.
- Callbacks (`DataStorageCallback`).
- CUDA's library workspaces (cuFFT plans, cuBLAS) and the fragmentation of PyTorch's cache beyond the allowance above.

Some estimates are well above what was measured. They are left so: they are on the safe side.
The measured peaks are those of the whole tutorial (Windows, RTX 5090, 10 Oct 2026), which also holds everything
else the tutorial keeps. Estimate against measured, RAM / GPU in GB:

| Estimate | Tutorial | Estimate | Measured (whole tutorial) |
|---|---|---|---|
| single scatter simulation, non-TOF sinogram | t_PETGATE_scat_sino | 25.0 / 5.2 | 23.7 / 1.7 |
| single scatter simulation, 21 TOF bins, lazy | t_PETGATE_scat_sinoTOF | 21.4 / 6.8 | 19.8 / 5.6 |
| single scatter simulation, list mode | t_PETGATE_scat_lm | 29.5 / 5.2 | 22.1 / 3.2 |
| single scatter simulation, list mode, 21 TOF bins | t_PETGATE_scat_lmTOF | 24.9 / 4.8 | 22.9 / 5.2 |
| single scatter simulation, list mode | t_PETSIRD | 28.7 / 6.2 | 21.9 / 2.9 |
| StarGuide OSEM 10 x 10 | t_starguide | 8.1 / 3.4 | 5.3 / 1.1 |
| SPECT with a PSF operator, OSEM 100 x 3 | t_ac225_dicom_recon | 4.8 / 0.9 | 4.9 / 0.5 |
| SPECT with a PSF operator, OSEM 50 x 4 | t_ac225_simind_recon | 4.9 / 0.9 | 4.3 / 0.5 |

The single scatter simulation's estimate is above the peak of the whole tutorial, of which it is one step.
StarGuide's estimate is about 1.5 times the RAM and 3 times the GPU memory measured.
With a PSF operator, the GPU estimate is about twice what was measured.
