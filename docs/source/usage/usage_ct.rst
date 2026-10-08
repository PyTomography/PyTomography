************************
CT
************************

++++++++++++++++++++
DICOM-CT-PD
++++++++++++++++++++

PyTomography reads scans in the open DICOM-CT-PD format with :func:`pytomography.io.CT.dicom_ct_pd.get_projections_and_metadata_gen3`, which puts the views in acquisition order and, by default, filters photon-starved rays (the streaks between the shoulders, for example; ``low_signal_filter=False`` turns this off). Scanner conventions that a comparison with one scanner's own images suggested (detector centre, angle, table feed, a per-column calibration) are options that are off by default.

A scan can be reconstructed analytically with :class:`pytomography.algorithms.FilteredBackProjection`, which for these scanners performs helical weighted filtered back projection (WFBP), with or without a flying focal spot, within a GPU memory budget; or iteratively, for example with OS-SART as in the tutorial below, which can start from the FBP image (``object_initial``).

.. code-block:: python

    from pytomography.io.CT import dicom_ct_pd
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.metadata import ObjectMeta
    from pytomography.projectors.CT import CTGen3SystemMatrix

    proj, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3('path/to/projections')
    system_matrix = CTGen3SystemMatrix(ObjectMeta(dr=(1, 1, 1), shape=(512, 512, 300)), proj_meta)
    image = FilteredBackProjection(proj, system_matrix, filter='hann', slice_thickness=1.25)()
    hu = 1000 * (image / proj_meta.water_attenuation - 1)

.. grid:: 1 2 3 3
    :gutter: 2
    
    .. grid-item-card:: CT DICOM-CT-PD 
        :link: ../notebooks/t_CT_GEN3
        :link-type: doc
        :link-alt: PETGATE Scatter Sinogram tutorial
        :text-align: center

        :material-outlined:`wifi;4em;sd-text-secondary`

.. toctree::
    :maxdepth: 1
    :hidden:

    ../notebooks/t_CT_GEN3