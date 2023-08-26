:py:mod:`pytomography.utils.fourier_filters`
============================================

.. py:module:: pytomography.utils.fourier_filters


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.utils.fourier_filters.RampFilter
   pytomography.utils.fourier_filters.HammingFilter




.. py:class:: RampFilter

   Implementation of the Ramp filter :math:`\Pi(\omega) = |\omega|`


   .. py:method:: __call__(w)



.. py:class:: HammingFilter(wl, wh)

   Implementation of the Hamming filter given by :math:`\Pi(\omega) = \frac{1}{2}\left(1+\cos\left(\frac{\pi(|\omega|-\omega_L)}{\omega_H-\omega_L} \right)\right)` for :math:`\omega_L \leq |\omega| < \omega_H` and :math:`\Pi(\omega) = 1` for :math:`|\omega| \leq \omega_L` and :math:`\Pi(\omega) = 0` for :math:`|\omega|>\omega_H`. Arguments ``wl`` and ``wh`` should be expressed as fractions of the Nyquist frequency (i.e. ``wh=0.93`` represents 93% the Nyquist frequency).


   .. py:method:: __call__(w)



