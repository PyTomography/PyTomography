# PyTomography

Welcome to Pytomography, a package that is extremely fast a SPECT reconstruction. This library currently supports attenuation correction, PSF modeling, and Bayesian priors. As long as your data can be turned into a `torch.tensor`, it can be reconstructed. Here, for example, is reconstruction on a SIMIND projection output file.

```
osem_net = get_osem_net(projections_header = '/home/ubuntu/test_files/projections.h00',
                        CT_header = '/home/ubuntu/test_files/CT.hct',
                        psf_meta=PSFMeta(collimator_slope=0.0301, collimator_intercept=0.00197),
                        prior = LogCoshPrior(beta=0.05),
                        device=device) 
reconstructed_object= osem_net(n_iters=4, n_subsets=8)                 
```

![](/images/sample_MIP.png)

# Installation

This library requires use of the pytorch machine learning library. I recommend making a virtual environment using anaconda:

```
conda create --name pytomography
```

Then install whatever version of pytorch you need using the commands [here](https://pytorch.org/get-started/locally/).

Finally, install pytomography:

```
pip install pytomography
```

# More Reading

For an extensive overview of this package, see the documentation.
