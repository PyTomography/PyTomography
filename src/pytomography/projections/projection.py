import torch.nn as nn
import abc
from pytomography.corrections import CorrectionNet
from pytomography.metadata import ObjectMeta, ImageMeta

class ProjectionNet(nn.Module):
    r"""Abstract parent class for projection networks. Any subclass of this network must implement the ``forward`` method. """
    def __init__(
        self,
        object_correction_nets: list[CorrectionNet],
        image_correction_nets: list[CorrectionNet],
        object_meta: ObjectMeta,
        image_meta: ImageMeta,
        device: str = 'cpu'
    ) -> None:
        """Initializer

        Args:
            object_correction_nets (list): Sequence of correction networks which operate on an object.
            image_correction_nets (list): Sequence of correction networks which operate on an image.
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
        """
        super(ProjectionNet, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets
        self.image_correction_nets = image_correction_nets
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.initialize_correction_nets()

    def initialize_correction_nets(self):
        """Function that initializes all correction networks with the required object and image metadata corresponding to the projection network.
        """
        for net in self.object_correction_nets:
            net.initialize_network(self.object_meta, self.image_meta)
        for net in self.image_correction_nets:
            net.initialize_network(self.object_meta, self.image_meta)

    @abc.abstractmethod
    def foward(self):
        """Abstract method that must be implemented by any subclass of this class.
        """
        ...