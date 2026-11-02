"""System matrix of proton CT: :class:`PCTSystemMatrix`, one row per proton along its estimated path (or a band of
paths when its entry is uncertain), and :func:`hull_from_image`, the object's outline from a first reconstruction."""
from .pct_system_matrix import PCTSystemMatrix, band_paths, hull_from_image
