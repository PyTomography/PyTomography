:py:mod:`pytomography.transforms.SPECT.psf`
===========================================

.. py:module:: pytomography.transforms.SPECT.psf


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.SPECT.psf.GaussianBlurNet
   pytomography.transforms.SPECT.psf.SPECTPSFTransform



Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.transforms.SPECT.psf.get_1D_PSF_layer



.. py:class:: GaussianBlurNet(layer_r, layer_z=None)

   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: forward(input)



.. py:function:: get_1D_PSF_layer(sigmas, kernel_size)

   Creates a 1D convolutional layer that is used for PSF modeling.

   :param sigmas: Array of length Lx corresponding to blurring (sigma of Gaussian) as a function of distance from scanner
   :type sigmas: array
   :param kernel_size: Size of the kernel used in each layer. Needs to be large enough to cover most of Gaussian
   :type kernel_size: int

   :returns: Convolutional neural network layer used to apply blurring to objects of shape [batch_size, Lx, Ly, Lz]
   :rtype: torch.nn.Conv2d


.. py:class:: SPECTPSFTransform(psf_meta)

   Bases: :py:obj:`pytomography.transforms.Transform`

   obj2obj transform used to model the effects of PSF blurring in SPECT. The smoothing kernel used to apply PSF modeling uses a Gaussian kernel with width :math:`\sigma` dependent on the distance of the point to the detector; that information is specified in the ``PSFMeta`` parameter.

   :param psf_meta: Metadata corresponding to the parameters of PSF blurring
   :type psf_meta: PSFMeta

   .. py:method:: configure(object_meta, image_meta)

      Function used to initalize the transform using corresponding object and image metadata

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta
      :param image_meta: Image metadata.
      :type image_meta: ImageMeta


   .. py:method:: compute_kernel_size(radius, axis)

      Function used to compute the kernel size used for PSF blurring. In particular, uses the ``min_sigmas`` attribute of ``PSFMeta`` to determine what the kernel size should be such that the kernel encompasses at least ``min_sigmas`` at all points in the object.

      :returns: The corresponding kernel size used for PSF blurring.
      :rtype: int


   .. py:method:: get_sigma(radius)

      Uses PSF Meta data information to get blurring :math:`\sigma` as a function of distance from detector.

      :param radius: The distance from the detector.
      :type radius: float

      :returns: An array of length Lx corresponding to blurring at each point along the 1st axis in object space
      :rtype: array


   .. py:method:: apply_psf(object, ang_idx)


   .. py:method:: forward(object_i, ang_idx)

      Applies the PSF transform :math:`A:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis.

      :param object_i: Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
      :type object_i: torch.tensor
      :param ang_idx: The projection indices: used to find the corresponding angle in image space corresponding to each projection angle in ``object_i``.
      :type ang_idx: int

      :returns: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
      :rtype: torch.tensor


   .. py:method:: backward(object_i, ang_idx, norm_constant = None)

      Applies the transpose of the PSF transform :math:`A^T:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis. Since the PSF transform is a symmetric matrix, its implemtation is the same as the ``forward`` method.

      :param object_i: Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
      :type object_i: torch.tensor
      :param ang_idx: The projection indices: used to find the corresponding angle in image space corresponding to each projection angle in ``object_i``.
      :type ang_idx: int
      :param norm_constant: A tensor used to normalize the output during back projection. Defaults to None.
      :type norm_constant: torch.tensor, optional

      :returns: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
      :rtype: torch.tensor



