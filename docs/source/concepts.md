# Concepts

This page collects the conventions and the mathematics that every part of PyTomography follows. Read it once before writing your own system matrix, transform or algorithm, and come back to it when shapes, units or angles look wrong.

## Coordinates

PyTomography uses the DICOM patient coordinate system: $x$ points from the patient's right to their left, $y$ from anterior to posterior, and $z$ from inferior to superior.

Index order follows coordinate order, and smaller indices are smaller coordinates. For an object tensor `f` of shape `(Lx, Ly, Lz)`, `f[-1, 0, 0]` is the voxel at the largest $x$ and the smallest $y$ and $z$.

### SPECT angles

The DICOM scanner angle $\beta$ is a counter-clockwise rotation from 12 o'clock, looking into the scanner. In the azimuthal angle $\phi$ of the coordinate system above, the detector sits at

$$\phi = \tfrac{3\pi}{2} - \beta ,$$

so for $\beta \in [0, 2\pi)$, $\phi \in [-\pi/2, 3\pi/2)$. To project at angle $\beta$, the SPECT system matrix rotates the object by $270^\circ - \beta$ so that the detector lies along $+x$, then sums along $x$. The detector's radial axis $r$ is aligned with $x$ at $\beta = 0$, which means it points along $-y$ at $\beta = 90^\circ$; this is the usual source of mirrored-looking projections.

```{image} images/coordinate_conventions.png
:alt: The x, y and z axes relative to the patient, the scanner angle beta, and the detector axes r and z
:width: 320px
```

## Tensors and units

Objects and projections are `torch.Tensor`s on `pytomography.device` (the GPU when one is available). There is no batch dimension: several energy windows or bed positions are handled as separate tensors or a leading index in the I/O functions that return them.

| | Object | Projections | Voxel and pixel sizes | Attenuation maps |
|---|---|---|---|---|
| SPECT | `(Lx, Ly, Lz)` | `(L_angles, L_r, L_z)` | cm | cm⁻¹ |
| PET, list mode | `(Lx, Ly, Lz)` | one value per event | mm | mm⁻¹ |
| PET, sinogram | `(Lx, Ly, Lz)` | `(L_angles, L_r, L_planes)`, with a TOF axis last when used | mm | mm⁻¹ |
| CT | `(Lx, Ly, Lz)` | `(L_views, L_columns, L_rows)` | mm | the reconstruction itself, in mm⁻¹ |

The metadata objects carry these sizes: `ObjectMeta` and `SPECTObjectMeta` describe the object grid, and the `ProjMeta` classes describe the projections.

## The system matrix

Every reconstruction in PyTomography is built on two operations of a **system matrix** $H$, where $H_{ij}$ is the mean contribution of voxel $j$ to detector element $i$:

- **Forward projection** maps an object $f$ to expected projections, $g = Hf$. It is the `forward` method of a `SystemMatrix`.
- **Back projection** maps projections to object space, $\hat f = H^T g$. It is the `backward` method. It must be the transpose of `forward`; the [contract tests](contributing/testing.md#contract-tests) check this for every system matrix.

$H$ is far too large to store: for a $128^3$ object and 64 projections of $128^2$ it would have $2\times 10^6 \times 10^6$ entries. Instead, PyTomography computes it on the fly as a sequence of simpler operations. Write the projections as a set of views $g = \sum_\theta g_\theta \otimes \hat\theta$, where a view $\theta$ is one detector angle in SPECT, or one angle and ring difference in PET. Then

$$H = \sum_\theta \Big(\prod_i B_i(\theta)\Big)\, P(\theta)\, \Big(\prod_i A_i(\theta)\Big) \otimes \hat\theta ,$$

where $P(\theta)$ projects the object onto view $\theta$, the $A_i(\theta)$ are **object-to-object transforms** and the $B_i(\theta)$ are **projection-to-projection transforms**. The transpose reverses the order and transposes each factor:

$$H^T = \sum_\theta \Big(\prod_{i,\ \text{reversed}} A_i^T(\theta)\Big)\, P^T(\theta)\, \Big(\prod_{i,\ \text{reversed}} B_i^T(\theta)\Big) \otimes \hat\theta^T .$$

- **SPECT** uses object-to-object transforms: attenuation, $A_2(\theta)$, depends on how much material lies between each voxel and the detector at angle $\theta$, and the collimator-detector response, $A_1(\theta)$, blurs each plane parallel to the detector by an amount that depends on its distance from the detector.
- **PET** applies attenuation in projection space, because the probability of detecting a coincidence is the same for every point along a line of response.

Transforms live in `pytomography.transforms`; you can add your own by subclassing `Transform`.

## Reconstruction algorithms

The measured projections $g$ are Poisson distributed with mean $Hf + s$, where $s$ is an additive term such as scatter (SPECT) or scatter and randoms (PET). Maximising the Poisson log-likelihood with expectation maximisation gives **MLEM**:

$$f^{(n+1)} = \frac{f^{(n)}}{H^T \mathbf{1}}\; H^T\!\left(\frac{g}{Hf^{(n)} + s}\right) .$$

**OSEM** splits the views into $M$ subsets $\Theta_0, \dots, \Theta_{M-1}$ and applies the same update with each subset's projections $g_m$ and system matrix $H_m$ in turn:

$$f^{(n,m+1)} = \frac{f^{(n,m)}}{H_m^T \mathbf{1}}\; H_m^T\!\left(\frac{g_m}{H_m f^{(n,m)} + s_m}\right), \qquad f^{(n,M)} \equiv f^{(n+1,0)} .$$

Each OSEM iteration costs about as much as one MLEM iteration but makes $M$ updates, so it converges much faster.

### Priors

A prior $V(f)$ with strength $\beta$ encodes what a plausible image looks like, for example that neighbouring voxels have similar values. It turns the likelihood into $L(f)\,e^{-\beta V(f)}$, and the gradient of $V$ must be evaluated somewhere in the update:

- **One step late** (`OSMAPOSL`) uses the previous estimate:
  $$f^{(n,m+1)} = \frac{f^{(n,m)}}{H_m^T\mathbf{1} + \beta \nabla V\big(f^{(n,m)}\big)}\; H_m^T\!\left(\frac{g_m}{H_m f^{(n,m)} + s_m}\right).$$
- **Block sequential regularised EM** (`BSREM`) takes an EM step, then a relaxed gradient step on the prior with step size $\alpha_n$:
  $$f^{(n,m+1)} = f^{(n,m+1/2)}\left(1 - \beta\,\frac{\alpha_n}{H_m^T\mathbf{1}}\,\nabla V\big(f^{(n,m+1/2)}\big)\right).$$

The algorithms live in `pytomography.algorithms`, the likelihoods in `pytomography.likelihoods` and the priors in `pytomography.priors`. The [algorithms tutorial](notebooks/t_algorithms.ipynb) compares them on the same data.
