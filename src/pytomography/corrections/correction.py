import abc
import torch.nn as nn

class CorrectionNet(nn.Module, metaclass=abc.ABCMeta):
    """Correction net is the parent class for all correction networks used in reconstruction. It must take in the object/image metadata, and the corresponding pytorch device used for computation

        Args:
            object_meta (ObjectMeta): Metadata for object space.
            image_meta (ImageMeta): Metadata for image space.
            device (str): Pytorch device used for computation
        """
    def __init__(self, object_meta, image_meta, device):
        super(CorrectionNet, self).__init__()
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.device = device
    @abc.abstractmethod
    def forward(self):
        """Must be implemented by the child class correction network
        """
        pass