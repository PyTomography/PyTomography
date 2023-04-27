# PyTomography

<p style='text-align: justify;'>
PyTomography is a flexible, high performance, open source package developed in Python&reg for medical image reconstruction. It uses the functions and classes of PyTorch to enable GPU-accelerated operations. The main focuses are

1.  Fast reconstruction of medical imaging data. 
2.  Providing a framework for the development of novel reconstruction algorithms.

## Currently Supported Modalities
* Single Photon Computed Emission Tomography (SPECT)
* Positron Emission Tomography (PET) (2D, no scatter). Implementation of full 3D PET with scatter is currently being developed.

## Example
This example uses generated simulated SPECT projections using the XCAT phantom and the SIMIND Monte Carlo code. Scatter, attenuation and PSF modeling are used to make the reconstructed images quantitative. In addition, the Bayesian relative difference prior.

```
reconstruction_algorithm = get_SPECT_recon_algorithm_simind(
    projections_header = 'body1t2ew6_tot_w2.hdr',
    scatter_headers = ['body1t2ew6_tot_w1.hdr',
                       'body1t2ew6_tot_w3.hdr'],
    CT_header = 'body1.hct',
    psf_meta=PSFMeta(collimator_slope=0.03013, collimator_intercept=0.1967),
    prior = RelativeDifferencePrior(beta=1, gamma=5),
    recon_algorithm_class=OSEMBSR)
reconstructed_object = reconstruction_algorithm(n_iters=10, n_subsets=8)                 
```

Maximum intensity projections corresponding to the reconstructed SPECT object above are shown below:
![](images/sample_MIPa.png)

## Installation

This library requires a local installation of PyTorch. As such, it is recommended to first create a virtual environment using anaconda:

```
conda create --name pytomography
```

and then install the version of PyTorch you need inside that environment [here](https://pytorch.org/get-started/locally/). Finally, install pytomography using the following command:

```
pip install pytomography
```

