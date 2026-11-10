"""Proton CT: a GATE simulation of a single-sided 330 MeV scan of an abdomen, a quarter of its protons (47 million)
(the data of t_pct_abdomen).

    python benchmarks/cases/pct_abdomen.py --recon fbp    # two-pass FBP (straight paths, then inside the outline)
    python benchmarks/cases/pct_abdomen.py --recon sart   # SART along curved paths: 4 passes, 180/180/60/20 subsets
"""
from _bench import Bench

from pytomography import datasets, io
from pytomography.algorithms import SART, FilteredBackProjection
from pytomography.projectors.PCT import PCTSystemMatrix, hull_from_image

bench = Bench('pct_abdomen', ['fbp', 'sart'], {
    'fbp': 'Proton CT (330 MeV, single-sided, 47 M protons), two-pass FBP with the outline (t_pct_abdomen)',
    'sart': 'Proton CT (330 MeV, single-sided, 47 M protons), SART along curved paths, 4 passes (t_pct_abdomen)'})

datasets.fetch('PCT/GATE-Abdomen330-DTC')
folder = datasets.data_dir() / 'PCT' / 'GATE-Abdomen330-DTC'

with bench.stage('load'):
    truth = io.load_image(folder / 'truth_rsp.nii.gz')
    wepl, object_meta, proj_meta = io.PCT.load_listmode(folder / 'abdomen330_spot_likely_quarter.h5',
                                                        dr=truth.voxel_size, shape=truth.shape)
    del truth


def two_pass_fbp():
    first = FilteredBackProjection(wepl, PCTSystemMatrix(object_meta, proj_meta, path='straight', band=5), filter='hann')()
    hull = hull_from_image(first, object_meta)
    second = FilteredBackProjection(wepl, PCTSystemMatrix(object_meta, proj_meta, path='straight', band=5, hull=hull),
                                    filter='hann', line='hull')()
    return first, hull, second


bench.setting(grid=list(object_meta.shape), voxel_mm=[round(float(d), 3) for d in object_meta.dr],
              protons=int(wepl.numel()))
if bench.recon == 'fbp':
    bench.setting(algorithm='FBP, two passes', window='hann')
    bench.repeat('FBP, two passes', two_pass_fbp)
else:
    passes = [(3, 180)] if bench.quick else [(3, 180), (3, 180), (3, 60), (5, 20)]
    bench.setting(algorithm='SART, curved paths', passes=[f'band {b}, {s} subsets' for b, s in passes])
    with bench.stage('FBP, two passes (the starting image)'):
        first, hull, second = two_pass_fbp()
        support = hull_from_image(first, object_meta, grow=2)
        start = second.clamp(min=0)
        del first, second

    def sart():
        image = start
        for band, n_subsets in passes:
            image = SART(PCTSystemMatrix(object_meta, proj_meta, band=band, hull=hull), wepl, object_initial=image,
                         relaxation_sequence=lambda _: 0.5, support=support)(n_iters=1, n_subsets=n_subsets)
        return image
    bench.repeat(f'SART, {len(passes)} passes', sart)
bench.save()
