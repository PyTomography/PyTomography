import numpy as np

dt = np.dtype([
    ('norm_factor', np.float32),
    ('c1', np.int32),
    ('c2', np.int32),
])
norm_data = np.fromfile('/disk1/etsi_scanner/norm.cdf', dtype=dt)
norm_2D = np.zeros((24000,24000))
norm_2D[norm_data['c1'],norm_data['c2']] = 1/norm_data['norm_factor']
idx = np.triu_indices(24000, k=1)
eta = norm_2D[*idx] + norm_2D.T[*idx]
np.save('/disk1/etsi_scanner/eta', eta)