import numpy as np
import torch
import pytomography
from pytomography.mappings import MapNet
from pytomography.metadata import ObjectMeta, ImageMeta, PSFMeta

class PETPSFNet(MapNet):
    def __init__(self, kerns, device: str = pytomography.device) -> None:
        super(PETPSFNet, self).__init__(device)
        self.kerns = kerns
        
    def initialize_network(self, object_meta: ObjectMeta, image_meta: ImageMeta) -> None:
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.construct_matrix()
        
    def construct_matrix(self):
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
        mode: str = 'forward_project',
	) -> torch.tensor:
        image = image.permute(0,1,3,2).unsqueeze(dim=-1)
        if mode=='forward_project':
            image = torch.matmul(self.PSF_matrix,image)
        else:
            # Tranpose multiplication
            image = torch.matmul(self.PSF_matrix.permute(0,1,2,4,3),image)
        image = image.squeeze(dim=-1).permute(0,1,3,2)
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