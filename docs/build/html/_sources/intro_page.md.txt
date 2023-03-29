# PyTomography

<p style='text-align: justify;'>
PyTomography is a flexible, high performance, open source package developed in Python&reg for medical image reconstruction. It uses the functions and classes of PyTorch to enable GPU-accelerated operations.

## Objectives
PyTomography aims at:
1.  Fast reconstruction of medical imaging data. 
2.  Development of novel reconstruction algorithms. 

## Currently supported medical imaging modalities
* Single Photon Computed Emission Tomography (SPECT)

# SPECT reconstruction example
In this example we have generated simulated projections using the XCAT phantom and the SIMIND Monte Carlo code. We apply scatter and attenuation correction to make the images quantitative. We also correct for the blurring of the imaging system. The data is reconstructed using the relative difference prior.

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

# Installation

This library requires a local installation of PyTorch. As such, I recommend first creating a virtual environment using anaconda:

```
conda create --name pytomography
```

and then installing the version of PyTorch you need inside that environment [here](https://pytorch.org/get-started/locally/). Finally, install pytomography using the following command:

```
pip install pytomography
```

