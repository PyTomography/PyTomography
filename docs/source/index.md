% PyTomography documentation master file, created by
% sphinx-quickstart on Fri Feb  3 20:09:10 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.


# PyTomography
PyTomography is a python library for medical image reconstruction. It uses the functionality of PyTorch to (i) enable fast GPU-accelerated reconstruction and (ii) permit easy integration of deep-learning models in traditional reconstruction algorithms. **If you use PyTomography in your own research, please cite the following:** [https://arxiv.org/abs/2309.01977](https://arxiv.org/abs/2309.01977)

## Features
**Supported Modalities**
* Single Photon Computed Emission Tomography (SPECT)
    - System matrix modeling includes attenuation correction, PSF modeling, scatter correction
* 2D Positron Emission Tomography (PET)
    - System matrix modeling includes attenuation correction and radially dependent PSF modeling.

**Reconstruction Algorithms**
* Filtered Back Projection (FBP)
* Statistical Iterative Algorithms
    - OSEM / MLEM
    - OSEMOSL (see [here](https://ieeexplore.ieee.org/document/52985))
    - BSREM (see [here](https://ieeexplore.ieee.org/document/1207396))
    - KEM (see [here](https://ieeexplore.ieee.org/abstract/document/6868314))
    
Options exist to include anatomical information (such as MRI/CT) when using priors/regularization.

**Supported Datatypes**
* DICOM
    - Ability to open and align SPECT/CT data and create attenuation maps
    - Repository of collimator parameters for different scanners for obtaining PSF information
* SIMIND output files (interfile)
    - Functionality to combine multiple sets of projections (representing different organs/regions) into a single set of projection data

## Installation

This library requires a local installation of PyTorch. As such, it is recommended to first create a virtual environment using anaconda:

```
conda create --name pytomography
```

and then install the version of PyTorch you need inside that environment [here](https://pytorch.org/get-started/locally/). Finally, install pytomography using the following command:

```
pip install pytomography
```

Be sure to check out  {doc}`usage` for some simple examples. If you wish to make a contribution, please read the {doc}`developers_guide`.

## Examples
**Example 1**: *SPECT/CT Images of patient receiving targeted radionuclide therapy with Lu177 DOTATATE for neuroendocrine tumours (4 cycles of 7.4 GBq/cycle administered every 8 weeks). Row 1: OSEM (5 iterations, 8 subsets); row 2: BSREM (30 iterations, 8 subsets) using the RDP prior; row 3: BSREM (30 iterations, 8 subsets) using the RDP prior with anatomical weighting.*

![](images/deep_blue.jpg)

**Example 2**: *Reconstructed SIMIND maximum intensity projections corresponding to activity distribution of typical patient treated with with Lu177 DOTATATE. Ground truth was generated using the XCAT phantom software. Shown are reconstructions with OSEM (8 iterations and 120 iteratons), BSREM with the RDP prior (120 iterations), and BSREM with the RDP prior and anatomical weighting (120 iterations). SPECT projections are overlaid with MIPs of the attenuation coefficient (hence the visible skeleton, even though it had no sigificant uptake).*

<img src="https://drive.google.com/uc?export=view&id=1rTFjWTvTvxsBs34yo9Nc2urD-M4T4x_l">

## Contents

```{toctree}
:maxdepth: 1

usage
developers_guide
external_data
```

