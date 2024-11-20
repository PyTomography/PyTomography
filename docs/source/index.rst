************************
PyTomography
************************

**Useful links:** `Installation <install.html>`_ | `Tutorials <usage/usage.html>`_ | `API Reference <autoapi/index.html>`_ 

``PyTomography`` is a python library for tomographic reconstruction of medical imaging data. It provides a modularized framework for the construction and development of system matrices, likelihoods, and reconstruction algorithms.

.. note::
     If you use PyTomography in your research, please cite `the corresponding research paper <https://arxiv.org/abs/2309.01977>`_.

++++++++++++++++
Getting Started
++++++++++++++++

.. grid:: 1 2 3 3
    :gutter: 2

    .. grid-item-card:: Tutorials
        :link: tutorial-index
        :link-type: ref
        :link-alt: Tutorials
        :text-align: center

        :material-outlined:`accessibility_new;8em;sd-text-secondary`

        **New users start here!**
        Jupyter notebooks demonstrating how to reconstruct data from various file formats.

    .. grid-item-card:: Image Gallery
        :link: gallery-index
        :link-type: ref
        :link-alt: Tutorials
        :text-align: center

        :material-outlined:`photo;8em;sd-text-secondary`

        View some sample images reconstructed with PyTomography!

    .. grid-item-card:: Get Help
        :text-align: center

        :material-outlined:`live_help;8em;sd-text-secondary`

        .. button-link:: https://github.com/qurit/PyTomography/issues
            :shadow:
            :expand:
            :color: warning

            **Report an issue**

        .. button-link:: https://pytomography.discourse.group/
            :shadow:
            :expand:
            :color: warning

            **Post on Discourse**

.. toctree::
    :maxdepth: 2
    :hidden:

    install
    usage/usage
    gallery
    external_data
    developers_guide