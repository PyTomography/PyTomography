# Installation

PyTomography needs Python 3.10 or newer and PyTorch 2.4 or newer.

```bash
conda create -n pytomography python=3.12
conda activate pytomography
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pytomography
```

The third line installs PyTorch for an NVIDIA GPU with CUDA 12.8. For another GPU or CUDA version, take the command from
[PyTorch's site](https://pytorch.org/get-started/locally/); without a GPU, use `pip install torch`. Check the install:

```bash
python -c "import torch, pytomography; print(pytomography.__version__, torch.cuda.is_available())"
```

## Extras

Only some tutorials need these.

| For | Install |
|---|---|
| PET and CT | parallelproj 2, from conda-forge: `conda install -c conda-forge parallelproj-core`. On a GPU, add the build for your CUDA, e.g. `"libparallelproj=*=cuda130*"`. |
| Proton CT | CuPy for your CUDA: `pip install cupy-cuda12x` (CUDA 12) or `pip install cupy-cuda13x` (CUDA 13). |
| spectarf on an NVIDIA GPU | `pip install "spectarf[cuda]" "nvidia-cuda-nvcc-cu12==12.8.*"`, or `==12.6.*` with PyTorch for CUDA 12.6. On the CPU, spectarf needs nothing more. |
| SPECT hybrid Monte Carlo | [SIMIND 8.0](https://www.msf.lu.se/en/research/simind-monte-carlo-program/downloads): its folder first on `PATH`, and `SMC_DIR` set to its `smc_dir` folder, ending in a path separator. |

:::{dropdown} Supported NVIDIA GPUs

| GPU generation | Examples | PyTorch builds |
|---|---|---|
| Blackwell | RTX 50 series, RTX PRO 6000 Blackwell, B200 | cu128, cu130 |
| Hopper | H100, H200 | cu126, cu128, cu130 |
| Ada Lovelace | RTX 40 series, RTX 6000 Ada, L4, L40S | cu126, cu128, cu130 |
| Ampere | A100, RTX 30 series, RTX A6000, A10, A40 | cu126, cu128, cu130 |
| Turing | RTX 20 series, GTX 16 series, T4 | cu126, cu128, cu130 |
| Volta | V100, Titan V | cu126, cu128 |
| Pascal, Maxwell | GTX 10 and 900 series, P100, Tesla M40 | cu126 |

On Windows, a plain `pip install torch` gives the CPU-only build. AMD (ROCm, on Linux) and Apple silicon (MPS) GPUs run
PyTorch and PyTomography; spectarf's fast engine needs an NVIDIA GPU.
:::

:::{dropdown} spectarf: the command for your PyTorch's CUDA version

Find the version with `python -c "import torch; print(torch.version.cuda)"`, then:

| PyTorch's CUDA | Command |
|---|---|
| 13.0 or 12.8 | `pip install "spectarf[cuda]" "nvidia-cuda-nvcc-cu12==12.8.*"` |
| 12.6 | `pip install "spectarf[cuda]" "nvidia-cuda-nvcc-cu12==12.6.*"` |
| another 12.x | `pip install "spectarf[cuda]" "nvidia-cuda-nvcc-cu12==12.x.*"`, with your `x` (`12.8.*` for 12.9) |

The second package keeps spectarf's CUDA compiler at 12.8 or older: the 12.9 compiler rejects one of its kernels
(`nvvmError: ERROR_INVALID_IR`). On the CPU spectarf simulates a camera in about half an hour; on an RTX 5090 in about
1.5 minutes. It keeps what it simulates in a cache, so later runs take seconds.
:::

:::{dropdown} parallelproj: if Python stops with `OMP: Error #15`

Install `parallelproj-core`, not the full `parallelproj`: the full package makes conda install its own NumPy and SciPy,
whose OpenMP runtime clashes with PyTorch's. If you already see the error, run
`conda install -c conda-forge "libblas=*=*openblas"` in the environment. SPECT never needs parallelproj.
:::

:::{dropdown} SIMIND: why 8.0
PyTomography is tested with SIMIND 8.0. SIMIND 9.0 is not supported yet: its first build gives wrong projections for
voxel phantoms. PyTomography runs SIMIND itself; nothing else is needed.
:::

## Install for development

```bash
git clone https://github.com/PyTomography/PyTomography.git
cd PyTomography
pip install -e ".[dev]"
pytest
```

The [contributing guide](contributing/index.md) explains the rest. Coming from PyTomography 3?
[Moving from v3 to v4](migration.md) lists what changed.
