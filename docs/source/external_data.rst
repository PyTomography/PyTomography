************************
Data Tables
************************
PyTomography requires using a variety of data sources to obtain parameters required for tasks like PSF modeling in SPECT and converting CT data to attenuation maps. 

.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card:: NIST
        :link: https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients
        :link-alt: NIST
        :text-align: center

        :material-outlined:`biotech;8em;sd-text-secondary`

        PyTomography uses data from the national institute of standards and technology (NIST) to get attenuation coefficients for different elemental media and compounds.

    .. grid-item-card:: SPECT Collimator Data
        :link: collimator-data-index
        :link-type: ref
        :link-alt: Tutorials
        :text-align: center

        :material-outlined:`apps;8em;sd-text-secondary`

        Link to the available collimator codes and options for SPECT reconstruction in PyTomography.

.. toctree::
    :maxdepth: 1
    :hidden:

    data_tables/collimator_data

