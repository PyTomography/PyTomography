from __future__ import annotations
import numpy as np
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ImageMeta, PSFMeta

class PETPSFTransform(Transform):
    r"""im2im transform used to model the effects of PSF blurring in PET. The smoothing kernel is assumed to be independent of :math:`\theta` and :math:`z`, but is dependent on :math:`r`. 

    Args:
        kerns (Sequence[callable]): A sequence of PSF kernels applied to the Lr dimension of the image with shape [batch_size, Lr, Ltheta, Lz]
    """
    def __init__(self, kerns) -> None:
        super(PETPSFTransform, self).__init__()
        self.kerns = kerns
        
    def configure(self, object_meta: ObjectMeta, image_meta: ImageMeta) -> None:
        """Function used to initalize the transform using corresponding object and image metadata

        Args:
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
        """
        super(PETPSFTransform, self).configure(object_meta, image_meta)
        self.construct_matrix()
        
    def construct_matrix(self):
        """Constructs the matrix used to apply PSF blurring.
        """
        Lr = self.image_meta.padded_shape[1]
        dr = self.object_meta.dr[0]
        R = self.image_meta.radii[0]
        r = ((torch.arange(Lr) - Lr/2)*dr).unsqueeze(dim=1) + dr/2
        _, xv = torch.meshgrid(torch.arange(Lr*1.0), torch.arange(Lr)*dr)
        xv = xv - (torch.arange(Lr)*dr).unsqueeze(dim=1)
        self.PSF_matrix = torch.eye(Lr)
        for kern in self.kerns:
            M = torch.zeros((Lr,Lr))
            for i in range(Lr):
                if torch.abs(r[i]) < R:
                    M[i] = kern(xv[i],r[i],R)   
            self.PSF_matrix = self.PSF_matrix @ M
        self.PSF_matrix = self.PSF_matrix.reshape((1,1,1,*self.PSF_matrix.shape)).to(self.device)
    
    @torch.no_grad()
    def forward(
		self,
		image: torch.Tensor,
	) -> torch.tensor:
        r"""Applies the forward projection of PSF modeling :math:`B:\mathbb{V} \to \mathbb{V}` to a PET image.

        Args:
            image (torch.tensor]): Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the image

        Returns:
            torch.tensor: Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the PSF corrected image.
        """
        image = image.permute(0,1,3,2).unsqueeze(dim=-1)
        image = torch.matmul(self.PSF_matrix,image)
        return image
    
    @torch.no_grad()
    def backward(
		self,
		image: torch.Tensor,
        norm_constant: torch.Tensor | None = None,
	) -> torch.tensor:
        r"""Applies the back projection of PSF modeling :math:`B^T:\mathbb{V} \to \mathbb{V}` to a PET image.

        Args:
            image (torch.tensor]): Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the image
			norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

        Returns:
            torch.tensor: Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the PSF corrected image.
        """
        image = image.permute(0,1,3,2).unsqueeze(dim=-1)
        # Tranpose multiplication
        image = torch.matmul(self.PSF_matrix.permute(0,1,2,4,3),image)
        image = image.squeeze(dim=-1).permute(0,1,3,2)
        if norm_constant is not None:
            norm_constant = norm_constant.permute(0,1,3,2).unsqueeze(dim=-1)
            # Tranpose multiplication
            norm_constant = torch.matmul(self.PSF_matrix.permute(0,1,2,4,3),norm_constant)
            norm_constant = norm_constant.squeeze(dim=-1).permute(0,1,3,2)
            return image, norm_constant
        else:
            return image
    
def kernel_noncol(x,r,R, delta=1e-8):
    if r**2<R**2:
        sigma = torch.sqrt(R**2 - r**2)/4 * np.pi / 180
    else:
        sigma = torch.zeros(r.shape) + delta
    result = torch.exp(-x**2/sigma**2 / 2)
    return result / (torch.sum(result)+delta)

def kernel_penetration(x,r,R,mu=0.87, delta=1e-8):
    result = torch.exp(-torch.abs(mu*x / ((r/R)*torch.sqrt(1-(r/R)**2) + delta)))
    if r>=0:
        result*= x <= 0
    else:
        result*= x >= 0
    return result / (torch.sum(result)+delta)

def kernel_scattering(x,r,R,scatter_fact=0.327, delta=1e-8):
    sigma = scatter_fact * torch.sqrt(1-(r/R)**2) / (2 * np.sqrt(2*np.log(2))) # fwhm -> sigma
    result = torch.exp(-x**2/sigma**2 / 2)
    return result / (torch.sum(result)+delta)