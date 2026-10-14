"""Callbacks: code a reconstruction algorithm runs after each subiteration.

A callback can compute a metric of the image as the reconstruction goes (e.g. the noise in the liver at every
iteration of OSEM, or a noise-bias curve against a ground truth), or keep what uncertainty estimates need. Pass one
to the algorithm's call (``callback=``); its ``run`` method is called after each subiteration. User-defined callbacks
inherit from :class:`Callback`.

* :class:`DataStorageCallback`: keeps the object and expected projections of every subiteration, for
  ``compute_uncertainty``;
* :class:`MultiCallback`: several callbacks as one;
* :class:`Callback`: the base class.
"""
from .callback import Callback, MultiCallback
from .data_saving import DataStorageCallback