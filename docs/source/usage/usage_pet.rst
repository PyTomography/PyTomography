************************
PET
************************

+++++++++++++
GATE Data
+++++++++++++

The following 4 tutorials demonstrate the full capabilities of PET reconstruction in PyTomography with GATE data of a simulated brain phantom. The tutorials cover sinogram/listmode and nonTOF/TOF reconstruction with scatter (nonTOF/TOF) and random correction.

.. grid:: 1 2 4 4
    :gutter: 2
    
    .. grid-item-card:: GATE Sinogram
        :link: ../notebooks/t_PETGATE_scat_sino
        :link-type: doc
        :link-alt: PETGATE Scatter Sinogram tutorial
        :text-align: center

        :material-outlined:`psychology;4em;sd-text-secondary`

    .. grid-item-card:: GATE TOF Sinogram
        :link: ../notebooks/t_PETGATE_scat_sinoTOF
        :link-type: doc
        :link-alt: PETGATE Scatter Sinogram TOF tutorial
        :text-align: center

        :material-outlined:`psychology;4em;sd-text-secondary`

    .. grid-item-card::  GATE Listmode
        :link: ../notebooks/t_PETGATE_scat_lm
        :link-type: doc
        :link-alt: PETGATE Scatter LM tutorial
        :text-align: center

        :material-outlined:`psychology;4em;sd-text-secondary`

    .. grid-item-card:: GATE TOF Listmode
        :link: ../notebooks/t_PETGATE_scat_lmTOF
        :link-type: doc
        :link-alt: PETGATE Scatter LM TOF tutorial
        :text-align: center

        :material-outlined:`psychology;4em;sd-text-secondary`

+++++++++++++++++++
Clinical Data from Discovery MI
+++++++++++++++++++

This tutorial demonstrates how to reconstruct data exported from the GE software Duetto (listmode enabled). 

.. grid:: 1 2 4 4
    :gutter: 2
    
    .. grid-item-card:: Discovery MI 
        :link: ../notebooks/t_GE_HDF5
        :link-type: doc
        :link-alt: GE HDF5 tutorial
        :text-align: center

        :material-outlined:`camera;4em;sd-text-secondary`


+++++++++++++++++++
AI-Based reconstruction
+++++++++++++++++++

This tutorial demonstrates how to use the Deep Image Prior (DIP) reconstruction algorithm for reconstruction of brain phantom used in the GATE tutorials, and includes the development of a PyTorch neural network.

.. grid:: 1 2 4 4
    :gutter: 2

    .. grid-item-card:: GATE Deep Image Prior
        :link: ../notebooks/t_PETGATE_DIP
        :link-type: doc
        :link-alt: PETGATE DIP tutorial
        :text-align: center

        :material-outlined:`psychology;4em;sd-text-secondary`



.. toctree::
    :maxdepth: 1
    :hidden:

    ../notebooks/t_PETGATE_scat_sino
    ../notebooks/t_PETGATE_scat_sinoTOF
    ../notebooks/t_PETGATE_scat_lm
    ../notebooks/t_PETGATE_scat_lmTOF
    ../notebooks/t_GE_HDF5
    ../notebooks/t_PETGATE_DIP