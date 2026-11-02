# Installation

PyTomography needs Python 3.10 or newer and PyTorch 2.4 or newer. A GPU is recommended but not required: everything also runs on the CPU.

## Install with pip

We recommend an environment separate from your system Python, for example with conda:

```bash
conda create -n pytomography python=3.12
conda activate pytomography
pip install pytomography
```

`pip install pytomography` also installs [spectarf](#spect-psfs-from-monte-carlo-spectarf).

## GPUs

PyTomography computes on the GPU through PyTorch, so install the PyTorch build for your GPU first, following the [PyTorch instructions](https://pytorch.org/get-started/locally/), then PyTomography. On Windows, a plain `pip install torch` gives the CPU-only build. For example, for CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pytomography
```

Which NVIDIA GPUs each PyTorch build supports (PyTorch 2.9), and spectarf's fast GPU engine on them:

| GPU generation | Compute capability | Examples | PyTorch builds | spectarf's fast engine |
|---|---|---|---|---|
| Blackwell | 10.0, 12.0 | RTX 50 series, RTX PRO 6000 Blackwell, B200 | cu128, cu130 | yes, tested on an RTX 5090 |
| Hopper | 9.0 | H100, H200 | cu126, cu128, cu130 | yes |
| Ada Lovelace | 8.9 | RTX 40 series, RTX 6000 Ada, L4, L40S | cu126, cu128, cu130 | yes |
| Ampere | 8.0, 8.6 | A100, RTX 30 series, RTX A6000, A10, A40 | cu126, cu128, cu130 | yes |
| Turing | 7.5 | RTX 20 series, GTX 16 series, T4 | cu126, cu128, cu130 | yes |
| Volta | 7.0 | V100, Titan V | cu126, cu128 | yes |
| Pascal | 6.0, 6.1 | P100, GTX 10 series | cu126 | yes |
| Maxwell | 5.0, 5.2 | GTX 900 series, Tesla M40 | cu126 | yes |

spectarf's GPU kernels compile and link for every one of these architectures; they have been run on an RTX 5090. An AMD GPU (ROCm, on Linux) or Apple silicon (MPS) runs PyTorch, and PyTomography uses it, but spectarf's fast engine needs an NVIDIA GPU: spectarf then runs its PyTorch engine, which is slower.

## PET and CT: install parallelproj 2

PET and CT reconstruction use [parallelproj](https://parallelproj.readthedocs.io/) for their projectors. PyTomography 4 needs **parallelproj 2**. It is distributed through conda-forge, not PyPI, so install it with conda in the same environment:

```bash
conda install -c conda-forge parallelproj
```

To use the GPU, pick the `libparallelproj` build that matches your CUDA version, for example:

```bash
conda install -c conda-forge "libparallelproj=*=cuda130*" parallelproj
```

SPECT reconstruction does not need parallelproj. Importing PyTomography never requires it; only `pytomography.projectors.PET` does, and it tells you how to install it if it is missing.

## SPECT hybrid Monte Carlo: SIMIND

`MonteCarloHybridSPECTSystemMatrix` runs the SIMIND Monte Carlo program for its forward projections. Download **SIMIND 8.0** from its [downloads page](https://www.msf.lu.se/en/research/simind-monte-carlo-program/downloads) and install it as its instructions say: put the SIMIND folder first on `PATH` and set `SMC_DIR` to its `smc_dir` folder, ending in a path separator. PyTomography is tested with SIMIND 8.0; SIMIND 9.0 is not supported yet, because its first build gives wrong projections for voxel phantoms. Nothing else is needed: PyTomography drives SIMIND itself.

## SPECT PSFs from Monte Carlo: spectarf

[spectarf](https://pypi.org/project/spectarf/), Stephen Graves' package, makes a SPECT camera's point spread functions (PSFs) from Monte Carlo angular response functions (ARFs), given a description of the camera: its collimator, crystal and energy windows. PyTomography's ARF tutorials use it, and `pip install pytomography` installs it. Its Monte Carlo runs on the CPU, or on any GPU through PyTorch; on an NVIDIA GPU its own CUDA engine is much faster (about 1.5 minutes per camera on an RTX 5090, against half an hour on a CPU). That engine is an extra:

```bash
pip install "spectarf[cuda]"
```

**Windows with PyTorch for CUDA 12** (`cu126` or `cu128`): that PyTorch brings its own CUDA linker, of its CUDA version, and the extra's CUDA compiler must not be newer. Install the matching compiler with it:

```bash
pip install "spectarf[cuda]" "nvidia-cuda-nvcc-cu12==12.8.*"   # 12.6.* for cu126
```

PyTorch for CUDA 13 (`cu130`) needs nothing more, nor does Linux, where pip installs the compiler and the linker together. To check, this prints `fused-cuda` when the fast engine works (a mismatched compiler fails here with an nvJitLink error):

```bash
python -c "from spectarf import Collimator, DetectorStack; from spectarf.engines import transport; print(transport(DetectorStack(Collimator(4.0, 2.0, 59.7), crystal_mm=9.525), 208.0, 0.0, 0.0, 10000, engine='fused-cuda').engine)"
```

## Check the installation

```python
import torch
import pytomography

print(pytomography.__version__)
print("GPU available:", torch.cuda.is_available())
```

## Install for development

To work on PyTomography itself, clone the repository and install it in editable mode with the development extras:

```bash
git clone https://github.com/PyTomography/PyTomography.git
cd PyTomography
pip install -e ".[dev]"
pytest
```

The [contributing guide](contributing/index.md) explains the rest.
