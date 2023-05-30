# Data Files
This repository contains three data files related to medical imaging and radiation physics. The files are:

1. HU_to_mu.csv
2. SPECT_collimator_parameters.csv
3. lead_attenuation_values.csv

# HU_to_mu.csv
This file provides a mapping between Hounsfield Units (HU) and linear attenuation coefficients (mu) for various materials commonly encountered in medical imaging. The Hounsfield Units measure the radiodensity of a material, while the linear attenuation coefficient represents how much the material attenuates X-ray or gamma-ray photons. The file is formatted as a CSV (Comma-Separated Values) file, with two columns:

HU: Hounsfield Units (numerical values)
mu: Linear attenuation coefficient (in cm$^-1$)

# SPECT_collimator_parameters.csv 
This file contains parameters related to Single-Photon Emission Computed Tomography (SPECT) collimators used in different scanners. SPECT is a nuclear medicine imaging technique that provides functional information about the body. The file is formatted as a CSV file, with four columns:
- camera_model: Name of the scanner or camera model
- collimator_name: Name of the collimator used in the respective scanner
- hole_diameter: Diameter of the collimator hole in centimeters (cm)
- hole_length: Length of the collimator septal in centimeters (cm)

These parameters are crucial for the design and optimization of SPECT imaging systems and can be used for simulating and analyzing the performance of different collimators.

# lead_attenuation_values.csv
This file contains energy and linear attenuation coefficient values for lead. The linear attenuation coefficient represents how much lead attenuates X-ray or gamma-ray photons at different energy levels. The data in this file has been retrieved from the physics.nist.gov website and covers an energy range of 100 keV to 600 keV. The file is formatted as a CSV file, with two columns:

- energy_keV: Energy level in kiloelectron volts (keV)
- linear_attenuation_coefficient: Linear attenuation coefficient of lead in cm$^-1$

These values are essential for calculations involving lead shielding and radiation attenuation in medical and industrial settings.

Please note that the data in these files should be used for reference purposes and should be validated and cross-checked against trusted sources when applied to specific applications.

# License
The data files are under the same license as the general repository. See the main README file.



