:py:mod:`pytomography.priors`
=============================

.. py:module:: pytomography.priors


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   smoothness/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.priors.QuadraticPrior
   pytomography.priors.LogCoshPrior




.. py:class:: QuadraticPrior(beta, delta=1, device='cpu')

   Bases: :py:obj:`SmoothnessPrior`

   Implentation of `SmoothnessPrior` where :math:`\phi` is the identity function


.. py:class:: LogCoshPrior(beta, delta=1, device='cpu')

   Bases: :py:obj:`SmoothnessPrior`

   Implementation of `SmoothnessPrior` where :math:`\phi` is the hyperbolic tangent function


