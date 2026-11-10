"""CT, helical: a clinical low-dose chest scan's full-dose projections, from a third-generation scanner (the data of
t_fbp_ct).

    python benchmarks/cases/ct_helical_fbp.py --recon fbp   # filtered back projection, Hann window, 1.25 mm slices, 512 x 512 x 316
"""
from _bench import Bench

from pytomography import datasets, io
from pytomography.algorithms import FilteredBackProjection
from pytomography.metadata import ObjectMeta
from pytomography.projectors.CT import CTGen3SystemMatrix

bench = Bench('ct_helical_fbp', ['fbp'], {
    'fbp': 'CT helical FBP, Hann window, 1.25 mm slices, 512 x 512 x 316 (LDCT chest C145, t_fbp_ct)'})

datasets.fetch('CT/ldct-c145')
folder = datasets.data_dir() / 'CT' / 'ldct-c145'

with bench.stage('load'):
    proj, object_meta, proj_meta = io.CT.read_dicom_ct_pd(
        folder / 'full_dose_projections', object_meta=ObjectMeta(dr=(0.662109, 0.662109, 1.0), shape=(512, 512, 316)))

with bench.stage('system matrix'):
    system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)

bench.setting(grid=list(object_meta.shape), voxel_mm=[round(float(d), 4) for d in object_meta.dr],
              views=int(proj.shape[0]), detector=list(proj.shape[1:]), algorithm='FBP', window='hann', slice_thickness_mm=1.25)
bench.repeat('FBP (Hann)', lambda: FilteredBackProjection(proj, system_matrix, filter='hann', slice_thickness=1.25)())
bench.save()
