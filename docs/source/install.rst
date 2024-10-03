****************
Installation
****************

We recommend creating a python environment that is isolated from your system python installation. This can be done using `conda <https://docs.conda.io/en/latest/>`_, and creating a virtual environment via:

.. code-block:: bash

    conda create -n pytomography_env python=3.9
    conda activate pytomography_env

Once the environment is activated, you can install PyTomography via pip:

.. code-block:: bash

    pip install pytomography

+++++++++++++++
Additional Requirements (PET / CT)
+++++++++++++++

PET and CT reconstruction require use of `parallelproj <https://parallelproj.readthedocs.io/en/stable/>`_. Parallelproj can be installed in the same conda environment as PyTomography via:

.. code-block:: bash

    conda activate pytomography_env
    conda install -c conda-forge libparallelproj parallelproj cupy

Note that you must include the ``cupy`` package in the installation command, as it is required to run operations on GPU in parallelproj.