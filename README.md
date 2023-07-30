

# <img src="https://pytomography.readthedocs.io/en/latest/_static/PT1.png" alt= “” width="5%">  PyTomography    <img src="https://www.bccrc.ca/dept/io-programs/qurit/sites/qurit/files/FINAL_QURIT_PNG_60.png" alt= “” width="10%">

![Contributors](https://img.shields.io/github/contributors/qurit/PyTomography?style=plastic)
![Forks](https://img.shields.io/github/forks/qurit/PyTomography)
![Stars](https://img.shields.io/github/stars/qurit/PyTomography)
![Issues](https://img.shields.io/github/issues/qurit/PyTomography)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

## An open source library for quantitative medical imaging reconstruction.

### Why?

While the manufacturers of medical imaging equipment like Single Photon Emission Computed Tomography (SPECT) and Positron Emission Tomography (PET) scanners provide their own image reconstruction algorithms, the implementation details are not easily accessible to the users. 


### Objectives
PyTomography aims at creating an open source platform where the nuclear medicine community can easily contribute to the implementatino of novel image reconstruction algorithms in a way that is:

1. Open and transparent
2. Fast
3. Compatible with the DICOM standard

### The Vision
PyTomography will allow for the standardization of imaging protocols, improving the reliability, validity, and reproducibility of findings. This could ultimately help accelerate the development and translation of new diagnostic and therapeutic imaging capabilities into clinical practice.


## How to get started?

We have prepared a very detailed documentation guide available at [readthedocs](https://pytomography.readthedocs.io/en/latest/).

We cover details about:
* [Installation](https://pytomography.readthedocs.io/en/latest/#installation) 
* [Image Reconstruction 101](https://pytomography.readthedocs.io/en/latest/notebooks/conventions.html)
* [Tutorials](https://pytomography.readthedocs.io/en/latest/usage.html)
* [A Developers Guide](https://pytomography.readthedocs.io/en/latest/developers_guide.html)   
* [API Guide](https://pytomography.readthedocs.io/en/latest/autoapi/index.html)

## Some Technical Aspects
PyTomography uses the functionality of [PyTorch](https://pytorch.org/) to:
 1. Enable fast GPU-accelerated reconstruction 
 2. Permit easy integration of deep-learning models in traditional reconstruction algorithms.

It currently supports image reconstruction for SPECT/CT

It provides options to include Bayesian priors and anatomical information in those priors.

 
# License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
