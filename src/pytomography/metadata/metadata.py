class ObjectMeta():
    def __init__(self, dx, shape):
        self.dx = dx
        self.shape = shape

class ImageMeta():
    def __init__(self, object_meta, angles, radii=None):
        self.object_meta = object_meta
        self.angles = angles
        self.radii = radii
        self.num_projections = len(angles)
        self.shape = (self.num_projections, object_meta.shape[1], object_meta.shape[2])