.. _collimator-data-index:

++++++++++++
Collimator Codes
++++++++++++

Collimator data for SPECT imaging was obtained from the ``collim.col`` file of the SIMIND Monte Carlo program. The values in the "Code" column are used for the ``collimator_name`` argument of the ``get_psfmeta_from_scanner_params`` function from ``pytomography.io.SPECT``. The supported collimators/codes are listed below.

.. list-table:: Scanner Codes
   :widths: 25 25 50
   :header-rows: 1

   * - Scanner
     - Code
     - Comments
   * - Millenium VG kameran
     - GV-LEGP
     - LOW ENERGY GENERAL PURPOSE
   * - Millenium VG kameran
     - GV-LEHR
     - LOW ENERGY HIGH RESOLUTION
   * - Millenium VG kameran
     - GV-EEGP
     - EXTENDED LOW ENERGY GENERAL PURPOSE
   * - Millenium VG kameran
     - GV-MEGP
     - MEDIUM ENERGY GENERAL PURPOSE
   * - Millenium VG kameran
     - GV-HEGP
     - HIGH ENERGY GENERAL PURPOSE
   * - Millenium VG kameran
     - GV-LEUR
     - FAN-BEAM LOW-ENERGY ULTRA-HIGH RESOLUTION
   * - Millenium VG kameran
     - GV-UEUH
     - ULTRA-HIGH ENERGY ULTRA-HIGH RESOLUTION
   * - MPR kamera
     - GM-LEGP
     - LOW ENERGY GENERAL PURPOSE
   * - MPR kamera
     - GM-LEHR
     - LOW ENERGY HIGH RESOLUTION
   * - GE Infinia
     - GI-LEHR
     - Infinia High Resolution
   * - GE Infinia
     - GI-LEGP
     - Infinia General Purpose
   * - GE Infinia
     - GI-MEGP
     - Infinia MEGP
   * - GE Infinia
     - GI-HEGP
     - Infinia HEGP
   * - GE Infinia
     - GI-ELEG
     - Infinia Extended LEGP
   * - GE Infinia
     - GI-LEHX
     - Infinia High Resolution
   * - GE 870 DR
     - G8-LHRS
     - GE 870 DR High Resolution S
   * - GE 870 DR
     - G8-LEHS
     - GE 870 DR High Sensitivity
   * - GE 870 DR
     - G8-LUHR
     - GE 870 DR Ultra High Resolution
   * - GE 870 DR
     - G8-LEHR
     - GE 870 DR High Resolution
   * - GE 870 DR
     - G8-ELEG
     - GE 870 DR Extended General Purpose
   * - GE 870 DR
     - G8-MEGP
     - GE 870 DR Medium Energy
   * - GE 870 DR
     - G8-HEGP
     - GE 870 DR High Energy
   * - GE Ventri
     - GT-LEHR
     - Ventri High Resolution
   * - GE Ventri
     - GT-LEGP
     - Ventri General Purpose
   * - GE 640 Optima
     - GO-LEHR
     - Ventri High Resolution
   * - GE 640 Optima
     - GO-LEGP
     - Ventri General Purpose
   * - GE 630
     - GO-LEHR
     - Ventri High Resolution
   * - GE 630
     - GT-LEGP
     - Ventri General Purpose
   * - GE 670 CZT
     - GZ-LEHR
     - CZT Low Energy High Resolution
   * - CrystalCam CZT
     - HH-LEHR
     - CZT LE High Resolution (rect)
   * - CrystalCam CZT
     - HH-LEHS
     - CZT LE High Sensitivity (rect)
   * - CrystalCam CZT
     - HH-MEGP
     - CZT ME General Purpose (circ)
   * - BW = Philips Brightview camera
     - PB-LEGP
     - Philips Brightview LEGP
   * - BW = Philips Brightview camera
     - PB-LEHR
     - Philips Brightview LEHR
   * - BW = Philips Brightview camera
     - PB-CAHR
     - Philips Brightview Cadiac HR
   * - BW = Philips Brightview camera
     - PB-MEGP
     - Philips Brightview MEGP
   * - BW = Philips Brightview camera
     - PB-HEGP
     - Philips Brightview HEGP
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LEUR
     - LOW-ENERGY ULTRA HIGH RESOLUTION
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LEHR
     - LOW-ENERGY HIGH RESOLUTION
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LEGP
     - LOW-ENERGY GENERAL PURPOSE
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LEHS
     - LOW-ENERGY HIGH SENSITIVITY
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LEES
     - LOW-ENERGY EXTRA HIGH SENSITIVITY
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LESS
     - LOW-ENERGY SUPER HIGH SENSITIVITY
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LEUS
     - LOW-ENERGY ULTRA HIGH SENSITIVITY
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-LETY
     - LOW-ENERGY THYROID
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-ME23
     - MEDIUM-ENERGY I-123
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-ME19
     - MEDIUM-ENERGY 190 keV
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-ME26
     - MEDIUM-ENERGY 260 keV
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-ME30
     - MEDIUM-ENERGY 300 keV
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-ME36
     - MEDIUM-ENERGY 360 keV
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-METY
     - MEDIUM-ENERGY THYRO333ID
   * - VG = VON GAHLEN COLLIMATORS - HOLLAND
     - VG-HE51
     - HIGH 511 keV
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-LEUR
     - LOW-ENERGY ULTRA HIGH RESOLUTION
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-LEHR
     - LOW-ENERGY HIGH RESOLUTION
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-LEGP
     - LOW-ENERGY GENERAL PURPOSE
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-LEMS
     - LOW-ENERGY MEDIUM SENSITIVITY
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-LEHS
     - LOW-ENERGY HIGH SENSITIVITY
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-LEUS
     - LOW-ENERGY ULTRA HIGH SENSITIVITY
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-MEUS
     - MEDIUM-ENERGY ULTRA HIGH SENSITIVITY
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-MEHS
     - MEDIUM-ENERGY HIGH SENSITIVITY
   * - ENGINEERING DYNAMICS CORPORATION
     - ED-MEHR
     - MEDIUM-ENERGY HIGH RESOLUTION
   * - ME = MEDISO Nucline SPIRIT DH-V
     - ME-LEGP
     - Low-Energy General Purpose
   * - ME = MEDISO Nucline SPIRIT DH-V
     - ME-LEHR
     - Low-Energy High Resolution
   * - ME = MEDISO Nucline SPIRIT DH-V
     - ME-LEUHR
     - Low-Energy Ultra High Resolution
   * - ME = MEDISO Nucline SPIRIT DH-V
     - ME-MEGP
     - Medium Energy General Purpose
   * - ME = MEDISO Nucline SPIRIT DH-V
     - ME-HEGP
     - High Energy General Purpose
   * - MA = MEDISO AnyScan
     - MA-LEHS
     - Low-Energy General Purpose
   * - MA = MEDISO AnyScan
     - MA-LEHR
     - Low-Energy High Resolution
   * - MA = MEDISO AnyScan
     - MA-LEUHR
     - Low-Energy Ultra High Resolution
   * - MA = MEDISO AnyScan
     - MA-LHRHS
     - Low-Energy Ultra High Resolution
   * - MA = MEDISO AnyScan
     - MA-MEGP
     - Medium Energy General Purpose
   * - MA = MEDISO AnyScan
     - MA-HEGP
     - High Energy General Purpose
   * - MA = MEDISO AnyScan
     - MA-HLR
     - High Energy High Resolution
   * - GE Discovery NM630
     - D-LEGP
     - Low Energy General Purpose
   * - GE Discovery NM630
     - D-LEHR
     - Low Energy High Resolution
   * - GE Discovery NM630
     - D-MEGP
     - Medium Energy General Purpose
   * - GE Discovery NM630
     - D-HEGP
     - High Energy General Purpose
   * - GE Discovery NM630
     - D-UHEGP
     - Ultra High Energy General Purpose
   * - PR = PRISM 2000/3000 SYSTEMS
     - PR-HRFB
     - High Resolution Fan-Beam
   * - PR = PRISM 2000/3000 SYSTEMS
     - PR-URFB
     - UltraHigh Resol Fan-Beam
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-LEAP
     - Low-Energy All Purpose
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-LEHR
     - Low-Energy High Resolution
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-ME
     - Medium Energy Parallel
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-ME+
     - Medium Energy Parallel E.CAM+
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-HE
     - High Energy Parallel
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-HEGP
     - High Energy Parallel
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-511
     - 511 keV Collimator
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-512
     - 511 keV Collimator
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-THER
     - Therapy Collimator
   * - SI = SIEMENS MEDICAL SYSTEM
     - SI-DELU
     - Test
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-LEHS
     - Low-Energy High Sensitivity
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-LEAP
     - Low-Energy All Purpose
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-LEHR
     - Low-Energy High Resolution
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-LEUR
     - Low-Energy Ultra-High Resolution
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-LEFB
     - Low-Energy Fan-Beam
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-ME
     - Medium Energy
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-HE
     - High Energy
   * - SE = E.CAM SIEMENS MEDICAL SYSTEM
     - SE-UHE
     - Ultra-High Energy
   * - SY = SIEMENS MEDICAL SYSTEM SYMBIA COLLIMATORS
     - SY-LEAP
     - Low-Energy All Purpose
   * - SY = SIEMENS MEDICAL SYSTEM SYMBIA COLLIMATORS
     - SY-LEHR
     - Low-Energy High Resolution
   * - SY = SIEMENS MEDICAL SYSTEM SYMBIA COLLIMATORS
     - SY-ME
     - Medium Energy Parallel
   * - SY = SIEMENS MEDICAL SYSTEM SYMBIA COLLIMATORS
     - SY-HE
     - High Energy Parallel
   * - SY = SIEMENS MEDICAL SYSTEM SYMBIA COLLIMATORS
     - SY-HE1
     - High Energy Parallel
   * - SY = SIEMENS MEDICAL SYSTEM SYMBIA COLLIMATORS
     - SY-HE2
     - High Energy Parallel
