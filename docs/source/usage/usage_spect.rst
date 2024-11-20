************************
SPECT
************************

+++++++++++++++
Introduction
+++++++++++++++

The use cases below are basic introductions for how to use the library, and demonstrate complete reconstruction pipelines for SIMIND (Monte Carlo) and DICOM (clinical) phantom data using OSEM. They are a good starting point for learning the library.

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: SIMIND Introduction
        :link: ../notebooks/t_siminddata
        :link-type: doc
        :link-alt: SIMIND Intro tutorial
        :text-align: center

        :material-outlined:`accessibility_new;4em;sd-text-secondary`

    .. grid-item-card:: DICOM Introduction
        :link: ../notebooks/t_dicomdata
        :link-type: doc
        :link-alt: DICOM Intro tutorial
        :text-align: center

        :material-outlined:`add_circle;4em;sd-text-secondary`

+++++++++++++++
Available Algorithms
+++++++++++++++

The tutorials below demonstrates some of the available reconstruction algorithms of the library, and borrows code from the introduction tutorials above.

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: Various Algorithms
        :link: ../notebooks/t_algorithms
        :link-type: doc
        :link-alt: SIMIND Intro tutorial
        :text-align: center

        :material-outlined:`accessibility_new;4em;sd-text-secondary`

+++++++++++++++
Additional Use Cases
+++++++++++++++

The tutorials below demonstrate some additional use cases for SPECT reconstruction, such as reconstructing and stitching multi-bed positions, reconstructing multiple photopeaks, and estimating uncertainty in reconstructed images

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: Multi Bed Positions
        :link: ../notebooks/t_dicommultibed
        :link-type: doc
        :link-alt: multi bed
        :text-align: center

        :material-outlined:`bed;4em;sd-text-secondary`

    .. grid-item-card:: Multi Photopeak Reconstruction
        :link: ../notebooks/t_dualpeak
        :link-type: doc
        :link-alt: dual peak
        :text-align: center

        :material-outlined:`landscape;4em;sd-text-secondary`

    .. grid-item-card:: Uncertainty Estimation
        :link: ../notebooks/t_uncertainty_spect
        :link-type: doc
        :link-alt: uncertainty spect
        :text-align: center

        :material-outlined:`question_mark;4em;sd-text-secondary`

    .. grid-item-card:: Monte Carlo Scatter Correction (Lu-177)
        :link: ../notebooks/t_spect_mc
        :link-type: doc
        :link-alt: uncertainty spect
        :text-align: center

        :material-outlined:`swipe_right_alt;4em;sd-text-secondary`

    .. grid-item-card:: Monte Carlo Scatter Correction (Er-165)
        :link: ../notebooks/t_spect_mc2
        :link-type: doc
        :link-alt: uncertainty spect 2
        :text-align: center

        :material-outlined:`swipe_right_alt;4em;sd-text-secondary`

+++++++++++++++
Unique Systems
+++++++++++++++

Some SPECT systems have unique properties that require special handling. The tutorials below demonstrate how to handle these systems.

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: StarGuide
        :link: ../notebooks/t_starguide
        :link-type: doc
        :link-alt: multi bed
        :text-align: center

        :material-outlined:`star;4em;sd-text-secondary`


+++++++++++++++
Advanced PSF Models
+++++++++++++++

The tutorials below demonstrate how to use PSF models obtained via the `SPECTPSFToolbox <https://github.com/PyTomography/SPECTPSFToolbox>`_ for Ac-225 reconstruction, though they can be adapted for other isotopes.

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: Ac225 SIMIND
        :link: ../notebooks/t_ac225_simind_recon
        :link-type: doc
        :link-alt: simind
        :text-align: center

        :material-outlined:`offline_bolt;4em;sd-text-secondary`


    .. grid-item-card:: Ac225 DICOM
        :link: ../notebooks/t_ac225_dicom_recon
        :link-type: doc
        :link-alt: Ac225 DICOM
        :text-align: center

        :material-outlined:`offline_bolt;4em;sd-text-secondary`

+++++++++++++++
Cardiac Functionality
+++++++++++++++

The tutorials below demonstrate reconstruction and reorientation of cardiac SPECT data.

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: Cardiac Reconstruction and Reorientation
        :link: ../notebooks/t_CardiacReorientation
        :link-type: doc
        :link-alt: cardiac
        :text-align: center

        :material-outlined:`offline_bolt;4em;sd-text-secondary`

+++++++++++++++
Useful Code Snippets
+++++++++++++++

The following contains some useful code snippets you may require when working with PyTomography.

.. raw:: html

    <select id="code-select">
        <option value="base">Select Code Snippet...</option>
        <option value="multi-organ">Open multiple simind regions / energy windows at once</option>
        <option value="subsample-data">Resample projection data before reconstruction</option>
    </select>

.. literalinclude:: ../notebooks/p_simind_multiorgan.py
   :language: python
   :name: multi-organ
   :linenos:

.. literalinclude:: ../notebooks/p_subsample_data.py
   :language: python
   :name: subsample-data
   :linenos:

.. toctree::
    :maxdepth: 1
    :hidden:

    ../notebooks/t_siminddata
    ../notebooks/t_dicomdata
    ../notebooks/t_algorithms
    ../notebooks/t_dicommultibed
    ../notebooks/t_dualpeak
    ../notebooks/t_dicomuncertainty
    ../notebooks/t_ac225_simind_recon
    ../notebooks/t_ac225_dicom_recon