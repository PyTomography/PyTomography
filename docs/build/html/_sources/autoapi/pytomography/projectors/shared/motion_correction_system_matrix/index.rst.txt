:py:mod:`pytomography.projectors.shared.motion_correction_system_matrix`
========================================================================

.. py:module:: pytomography.projectors.shared.motion_correction_system_matrix


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.projectors.shared.motion_correction_system_matrix.MotionSystemMatrix




.. py:class:: MotionSystemMatrix(system_matrices, motion_transforms)

   Bases: :py:obj:`pytomography.projectors.system_matrix.ExtendedSystemMatrix`

   Abstract class for a general system matrix :math:`H:\mathbb{U} \to \mathbb{V}` which takes in an object :math:`f \in \mathbb{U}` and maps it to corresponding projections :math:`g \in \mathbb{V}` that would be produced by the imaging system. A system matrix consists of sequences of object-to-object and proj-to-proj transforms that model various characteristics of the imaging system, such as attenuation and blurring. While the class implements the operator :math:`H:\mathbb{U} \to \mathbb{V}` through the ``forward`` method, it also implements :math:`H^T:\mathbb{V} \to \mathbb{U}` through the `backward` method, required during iterative reconstruction algorithms such as OSEM.

   :param obj2obj_transforms: Sequence of object mappings that occur before forward projection.
   :type obj2obj_transforms: Sequence[Transform]
   :param im2im_transforms: Sequence of proj mappings that occur after forward projection.
   :type im2im_transforms: Sequence[Transform]
   :param object_meta: Object metadata.
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata.
   :type proj_meta: ProjMeta


