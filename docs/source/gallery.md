(gallery-index)=
# Gallery

Images made with PyTomography. [Share yours on Discourse](https://pytomography.discourse.group/) and we will add it here with a credit and a link to your paper.

## Figure 1: Lu-177 SPECT

A patient receiving Lu-177-PSMA-617. The images were reconstructed with MIM (left) and with four PyTomography algorithms. The noise in the liver and the total counts in selected regions are shown. The axial slices at the bottom show the region marked by the blue arrow, which indicates a bone metastasis in the sternum.

```{image} images/figure2.jpg
:alt: Lu-177 PSMA SPECT of a patient reconstructed with MIM and with four PyTomography algorithms
:width: 800px
```

## Figure 2: Ac-225 SPECT

A patient receiving Ac-225-PSMA-617. The PET scan on the left was not reconstructed with PyTomography. The right shows SPECT reconstructions at four time points after injection, made with the Monte Carlo PSF modelling of [SPECTPSFToolbox](https://spectpsftoolbox.readthedocs.io/en/latest/). The plots at the bottom show time-activity curves in three lesions marked on the PET image, with uncertainties from PyTomography's uncertainty estimation.

```{image} images/ac225_patient_dual.jpg
:alt: Ac-225 PSMA SPECT at four time points with time-activity curves and uncertainties
:width: 600px
```

## Figure 3: PET with Deep Image Prior

GATE Monte Carlo PET data of a brain phantom. Shown are the ground-truth PET/MR images, a high-count OSEM reconstruction, a low-count OSEM reconstruction, and a low-count Deep Image Prior reconstruction. Everything was done in PyTomography: TOF list-mode reconstruction, randoms and TOF scatter estimation, and building the network.

```{image} images/figure4_left_hoz.jpg
:alt: Brain PET phantom reconstructed with OSEM at high and low counts and with Deep Image Prior at low counts
:width: 800px
```

## Figure 4: Clinical CT

Clinical CT projections in DICOM-CT-PD format from a 3rd-generation scanner, reconstructed with OS-SART using 3 iterations and 40 subsets.

```{image} images/CT_slice.png
:alt: Axial slice of a clinical CT reconstructed with OS-SART
:width: 500px
```
