# PyTomography example scripts

Plain-Python versions of every tutorial: the code only, without plots or explanations, so you can read
a whole pipeline at a glance or run it as a script. Each file links to the full tutorial on the docs site.

These files are generated from the notebooks in `docs/source/notebooks` by `docs/tools/export_scripts.py`.
Edit the notebook, then run that script; CI checks the two match.

## SPECT

| Script | What it does |
|---|---|
| [`spect/01_simind_introduction.py`](spect/01_simind_introduction.py) | A complete OSEM pipeline on Monte Carlo data, from projections to a reconstructed image. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_siminddata.html) |
| [`spect/02_dicom_introduction.py`](spect/02_dicom_introduction.py) | Reconstruct a phantom exported from a clinical scanner and compare with the vendor image. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_dicomdata.html) |
| [`spect/03_reconstruction_algorithms.py`](spect/03_reconstruction_algorithms.py) | OSEM, BSREM, OSMAPOSL and KEM side by side on the same data. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_algorithms.html) |
| [`spect/04_accuracy_against_known_truth.py`](spect/04_accuracy_against_known_truth.py) | Measure bias and noise in every organ of a simulated patient, with OSEM and BSREM. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_accuracy_known_truth.html) |
| [`spect/05_multiple_bed_positions.py`](spect/05_multiple_bed_positions.py) | Reconstruct several bed positions and stitch them into one volume. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_dicommultibed.html) |
| [`spect/06_multiple_photopeaks.py`](spect/06_multiple_photopeaks.py) | Joint reconstruction over two photopeaks of the same isotope. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_dualpeak.html) |
| [`spect/07_uncertainty_estimation.py`](spect/07_uncertainty_estimation.py) | Voxel and region uncertainty that comes out of the reconstruction itself. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_uncertainty_spect.html) |
| [`spect/08_hybrid_monte_carlo_simulated.py`](spect/08_hybrid_monte_carlo_simulated.py) | Monte Carlo scatter estimated inside the reconstruction loop with SIMIND. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_spect_mc.html) |
| [`spect/09_hybrid_monte_carlo_measured.py`](spect/09_hybrid_monte_carlo_measured.py) | The same hybrid approach on measured data. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_spect_mc2.html) |
| [`spect/10_ge_starguide.py`](spect/10_ge_starguide.py) | A 12-head CZT system with body-contour sweeps. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_starguide.html) |
| [`spect/11_ac_225_psf_modelling_simind.py`](spect/11_ac_225_psf_modelling_simind.py) | Alpha-emitter imaging with a Monte Carlo collimator response, including penetration. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_simind_recon.html) |
| [`spect/12_ac_225_psf_modelling_dicom.py`](spect/12_ac_225_psf_modelling_dicom.py) | The Ac-225 PSF model applied to measured data. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_dicom_recon.html) |
| [`spect/13_cardiac_reorientation.py`](spect/13_cardiac_reorientation.py) | Reconstruct myocardial perfusion data and reorient it to the short axis. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_CardiacReorientation.html) |

## PET

| Script | What it does |
|---|---|
| [`pet/01_pet_introduction_list_mode_and_sinograms.py`](pet/01_pet_introduction_list_mode_and_sinograms.py) | Reconstruct the same true coincidences as list mode and as sinograms, with normalisation and attenuation. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_pet_introduction.html) |
| [`pet/02_gate_sinogram.py`](pet/02_gate_sinogram.py) | Non-TOF sinogram reconstruction of a simulated brain phantom with scatter and randoms. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_sino.html) |
| [`pet/03_gate_tof_sinogram.py`](pet/03_gate_tof_sinogram.py) | Time-of-flight sinogram reconstruction with TOF scatter estimation. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_sinoTOF.html) |
| [`pet/04_gate_list_mode.py`](pet/04_gate_list_mode.py) | List-mode reconstruction with scatter and randoms correction. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_lm.html) |
| [`pet/05_gate_tof_list_mode.py`](pet/05_gate_tof_list_mode.py) | Time-of-flight list-mode reconstruction with TOF scatter. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_lmTOF.html) |
| [`pet/06_ge_discovery_mi.py`](pet/06_ge_discovery_mi.py) | Clinical list mode exported from GE Duetto, with time of flight. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_GE_HDF5.html) |
| [`pet/07_pet_uncertainty_estimation.py`](pet/07_pet_uncertainty_estimation.py) | Uncertainty estimates for a clinical PET reconstruction. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_uncertainty_pet.html) |
| [`pet/08_petsird_list_mode.py`](pet/08_petsird_list_mode.py) | Read list-mode data in the open PETSIRD format. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_PETSIRD.html) |
| [`pet/09_deep_image_prior.py`](pet/09_deep_image_prior.py) | Build a PyTorch network and use it as the image model inside the reconstruction. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_DIP.html) |

## CT

| Script | What it does |
|---|---|
| [`ct/01_dicom_ct_pd.py`](ct/01_dicom_ct_pd.py) | OS-SART on raw projections from a 3rd-generation clinical CT scanner. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_GEN3.html) |

## Building with PyTomography

| Script | What it does |
|---|---|
| [`development/01_implementing_a_new_system_matrix.py`](development/01_implementing_a_new_system_matrix.py) | Write your own forward and back projector and reuse every algorithm in the library. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_examplesystemmatrix.html) |
| [`development/02_plotting_utilities.py`](development/02_plotting_utilities.py) | Fused SPECT/CT views and other figure helpers. [Tutorial](https://pytomography.readthedocs.io/en/latest/notebooks/t_plotting.html) |
