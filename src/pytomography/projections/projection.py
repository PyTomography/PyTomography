import torch.nn as nn
import abc
from pytomography.mappings import MapNet
from pytomography.metadata import ObjectMeta, ImageMeta

class ProjectionNet(nn.Module):
    r"""Abstract parent class for projection networks. Any subclass of this network must implement the ``forward`` method. """
    def __init__(
        self,
        obj2obj_nets: list[MapNet],
        im2im_nets: list[MapNet],
        object_meta: ObjectMeta,
        image_meta: ImageMeta,
        device: str = 'cpu'
    ) -> None:
        """Initializer

        Args:
            obj2obj_nets (list): Sequence of object mappings that occur before projection.
            im2im_nets (list): Sequence of image mappings that occur after projection.
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
        """
        super(ProjectionNet, self).__init__()
        self.device = device
        self.obj2obj_nets = obj2obj_nets
        self.im2im_nets = im2im_nets
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.initialize_correction_nets()

    def initialize_correction_nets(self):
        """Function that initializes all mapping networks with the required object and image metadata corresponding to the projection network.
        """
        for net in self.obj2obj_nets:
            net.initialize_network(self.object_meta, self.image_meta)
        for net in self.im2im_nets:
            net.initialize_network(self.object_meta, self.image_meta)

    @abc.abstractmethod
    def foward(self):
        """Abstract method that must be implemented by any subclass of this class.
        """
        ...