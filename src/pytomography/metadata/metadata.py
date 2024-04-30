from __future__ import annotations

class ObjectMeta():
    """Parent class for all different types of Object Space Metadata. In general, while this is fairly similar for all imaging modalities, required padding features/etc may be different for different modalities.
    """
    def __init__(self, dr, shape) -> None:
        self.dr = dr
        self.dx = dr[0]
        self.dy = dr[1]
        self.dz = dr[2]
        self.shape = shape
    
    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)


class ProjMeta():
    """Parent class for all different types of Projection Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.
    """
    def __init__(self, angles) -> None:
        self.angles = angles
        self.num_projections = len(angles)
    
    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)

        