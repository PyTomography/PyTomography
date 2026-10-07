# Installation

PyTomography needs Python 3.10 or newer and PyTorch 2.4 or newer. A GPU is recommended but not required: everything also runs on the CPU.

## Install with pip

We recommend an environment separate from your system Python, for example with conda:

```bash
conda create -n pytomography python=3.12
conda activate pytomography
pip install pytomography
```

If you want GPU support, install the PyTorch build that matches your CUDA version first, following the [PyTorch instructions](https://pytorch.org/get-started/locally/), then install PyTomography.

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
