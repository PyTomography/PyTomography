# PyTomography

PyTomography is a flexible, high performance python package for open source medical image reconstruction. It draws heavily from the functions and classes of PyTorch, permitting fast GPU-accelerated operations. It's main purposes are (i) fast reconstruction of imaging data and (ii) development of novel reconstruction algorithms. Here is an example of reconstruction on a SIMIND projection output file.

```
osem_net = get_osem_net(projections_header = 'body1t2ew6_tot_w2.hdr',
                        scatter_headers = ['body1t2ew6_tot_w1.hdr',
                                           'body1t2ew6_tot_w3.hdr'],
                        CT_header = 'body1.hct',
                        psf_meta=PSFMeta(collimator_slope=0.03013, collimator_intercept=0.1967),
                        prior = RelativeDifferencePrior(beta=1, gamma=5),
                        device=device)
reconstructed_object = osem_net(n_iters=10, n_subsets=8)                 
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

