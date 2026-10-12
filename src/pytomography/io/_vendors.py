"""The vendor table: every rule PyTomography applies because a vendor or a program writes its data its own way, with
the evidence for it.

Readers look a rule up with :func:`use`, which returns it, and warns (:class:`UnverifiedVendorRule`) when the rule is
not verified on data ("check": believed right, with a known open question; "unverified": no data to check it). The
table is also the documentation: :func:`table` lists every rule.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

#: Status values: verified on local data; believed right with an open question (to check in v4.1); not checked at all.
STATUSES = ('verified', 'check', 'unverified')


class UnverifiedVendorRule(UserWarning):
    """A vendor rule that has not been verified on data was used: the result may be misplaced. Please report a case
    where it is wrong (with the data's vendor, model and software version)."""


@dataclass(frozen=True)
class VendorRule:
    id: str
    vendor: str
    data: str
    rule: str
    evidence: str
    status: str

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f'{self.id}: status must be one of {STATUSES}')


_RULES = [
    VendorRule('ge-ct-image-corner', 'GE', 'CT images (DICOM)',
               'ImagePositionPatient is the corner of the first pixel, not its centre: add half a pixel along the row '
               'and column directions.',
               'TCIA LDCT C145: 0.325/0.330 mm measured against the projection data; Optima CT540 header arithmetic '
               'gives round centres only with this rule.', 'verified'),
    VendorRule('ge-ct-pd-central-column', 'GE', 'DICOM-CT-PD projections',
               'central column = n_columns - DetectorCentralElement[0]; rows as stored.',
               'C145 half-turn images coincide only with this value.', 'verified'),
    VendorRule('ge-nm-projection-ipp', 'GE', 'NM dual-head projections',
               'ImagePositionPatient (x, y) of the Detector Information Sequence is the centre of voxel (0, 0), as '
               'written; the rotation axis is at the centre of the projection columns.',
               'Lu177-PSMA-GEDisc: kidney masks from the CT land on the hot kidneys at 0 shift in whole 4.4 mm voxels; '
               'a half-pixel (2.2 mm) corner offset is not excluded.', 'check'),
    VendorRule('ge-starguide', 'GE', 'StarGuide NM projections and reconstructions',
               'Head geometry in private tags (0099,1051) x and (0099,1052) y in cm and (0099,1053) angle in degrees; '
               'frame times (0009,1003) in ms; stored values times the Pixel Scale (0011,103B); no '
               'ImagePositionPatient in the projections: the reconstruction grid is centred on the CT volume.',
               'PyTomography\'s grid matches GE\'s reconstruction geometry to < 0.01 mm; NCC 0.96 with GE\'s '
               'reconstruction, every flip <= 0.49.', 'verified'),
    VendorRule('ge-nm-mu-scale', 'GE', 'NM attenuation maps (DICOM)',
               'Values are stored values divided by the private scale (0033,1038).',
               'Code branch only; no local data.', 'unverified'),
    VendorRule('ge-dmi-listmode', 'GE', 'Discovery MI list mode (HDF5)',
               'id = 544 ring + crystal; TOF bin = 14 - raw/13; crystal 0 at the top; the bed posterior after a 180 '
               'degree turn about z.',
               'The TOF sign images the spheres; the bed position from the attenuation factors. x and z are not '
               'verified.', 'check'),
    VendorRule('siemens-nm-projection', 'Siemens', 'NM projections',
               'ImagePositionPatient of the Detector Information Sequence is (0, 0, z_top); the rotation axis is at '
               'y = -TableHeight.',
               "Lu177-NEMA-SymT2: PyTomography's reconstruction matches Siemens' own in patient space (NCC 0.955), "
               'every axis flip worse.', 'verified'),
    VendorRule('siemens-ct-image-y-origin', 'Siemens', 'CT images (DICOM)',
               "The patient y origin is the table: the image centre is at y = -TableHeight (ImagePositionPatient "
               "follows the standard, so images need no correction; raw data that refer to the isocentre do).",
               'Two Symbia CT series.', 'verified'),
    VendorRule('siemens-ct-pd-y-origin', 'Siemens', 'DICOM-CT-PD projections',
               'The patient y origin is unknown (the files carry no TableHeight).',
               'No local data (TCIA Siemens case in v4.1).', 'check'),
    VendorRule('siemens-mmr-mu-interfile', 'Siemens', 'mMR attenuation map (JSRecon Interfile .v.hdr)',
               "Keys 'matrix size[n]' and 'scale factor (mm/pixel) [n]'; x fastest; HFS; 'quantification units' "
               '1/cm; centred on the scanner.',
               'Placement against the PETSIRD mMR NEMA prompts in x, y and z.', 'verified'),
    VendorRule('mediso-nm-projection', 'Mediso', 'NM projections',
               'ImagePositionPatient x is voxel (0, 0); the rotation axis is at y = 0.',
               'Code branch only; no local data.', 'unverified'),
    VendorRule('mediso-ct-image', 'Mediso', 'CT images (DICOM)',
               'ImagePositionPatient y is relative to the table: subtract TableHeight.',
               'Code branch only; no local data.', 'unverified'),
    VendorRule('simind-units', 'SIMIND', 'headers (.h00, .hct)',
               "Radius and collimator keys in cm before version 8, in mm from version 8 on; the version is the "
               "number after 'V' in 'program version'.",
               'XCAT lungs; V7 and V8 radius both 15 cm.', 'verified'),
    VendorRule('simind-density-layout', 'SIMIND', 'density and attenuation maps (.hct/.ict)',
               "x fastest; the first slice is superior; the y direction is set by simind.ini index 22 ('mu': 3, "
               "'mu-castor': 5), which the header does not record, so the layout must be given; values in 1/cm; "
               "centred.",
               'XCAT (SIMIND-MultiOrgan): the lung sources sit on the lungs with layout 5 (0.71 and 0.78 of their '
               'counts), not with 3 (0.51).', 'verified'),
    VendorRule('simind-projection-frame', 'SIMIND', 'projections (.h00/.a00)',
               "x fastest within a view; row 0 at +z; angles from 'start angle', 'extent of rotation' and "
               "'direction of rotation'; the phantom centred on the rotation axis.",
               'Jaszak spheres and XCAT anatomy in LPS as read; counter-clockwise data from 0 only.', 'verified'),
    VendorRule('simind-clockwise-start', 'SIMIND', 'projections (.h00/.a00), clockwise with a non-zero start angle',
               "The angle of view n is -(start angle) - n * step for clockwise acquisitions.",
               "No local data; PyTomography's SIMIND writer and the reader disagree on the sign.", 'unverified'),
    VendorRule('gate-world-frame', 'GATE', 'world frame',
               'The patient is wherever the phantom was placed in the world: world_axes, world_origin.',
               "Hosted brain data: 'LPI', the MR's centre.", 'verified'),
    VendorRule('gate-stir-interfile-image', 'GATE/STIR', 'Interfile images (.hv/.v)',
               "x fastest, then y and z, as +x, +y and +z; centred on the scanner (GATE centres a voxelised phantom on "
               "its placement; STIR's 'first pixel offset' is relative to STIR's own origin at the first ring, so "
               "it is kept in meta and not used).",
               'The hosted water cube is symmetric, so the orientation cannot be checked.', 'unverified'),
    VendorRule('stir2petsird', 'STIR2PETSIRD', 'PETSIRD gantry frame',
               'x up, z towards the bed, the axial origin at the first ring; crystals placed by a corner (-0.353 '
               'degrees).',
               'The Siemens mu map and every crystal on its element.', 'verified'),
    VendorRule('nikon-xtek', 'Nikon', 'XTek cone beam',
               'Negate the .ang angles (the stage turns the sample); the centre of rotation is not in the file.',
               'Reprojection r 0.999; COR 0.24 mm by sharpness (the mirror cannot be checked).', 'verified'),
]

RULES = {rule.id: rule for rule in _RULES}


def use(rule_id: str, source: str = '') -> VendorRule:
    """The rule ``rule_id``, with a warning when it is not verified.

    Args:
        rule_id (str): the rule's id (see :data:`RULES`).
        source (str, optional): the file or data it is applied to, for the warning.

    Returns:
        VendorRule: the rule.
    """
    rule = RULES[rule_id]
    if rule.status != 'verified':
        where = f' ({source})' if source else ''
        warnings.warn(f'{rule.vendor} {rule.data}{where}: the rule "{rule.rule}" is used, but it is {rule.status} '
                      f'({rule.evidence}). Check the result, and please report a case where it is wrong.',
                      UnverifiedVendorRule, stacklevel=3)
    return rule


def table() -> list[dict]:
    """Every rule, as a list of dicts (vendor, data, rule, evidence, status)."""
    return [dict(id=r.id, vendor=r.vendor, data=r.data, rule=r.rule, evidence=r.evidence, status=r.status) for r in _RULES]


def manufacturer(ds) -> str:
    """The vendor of a DICOM dataset, from its Manufacturer: 'GE', 'SIEMENS', 'MEDISO', 'PHILIPS', 'CANON',
    'PYTOMOGRAPHY', or the Manufacturer itself (upper case) for others, '' when there is none."""
    name = str(getattr(ds, 'Manufacturer', '') or '').strip().upper()
    for key, vendor in (('GE ', 'GE'), ('GE MEDICAL', 'GE'), ('GE HEALTHCARE', 'GE'), ('SIEMENS', 'SIEMENS'),
                        ('MEDISO', 'MEDISO'), ('PHILIPS', 'PHILIPS'), ('CANON', 'CANON'), ('TOSHIBA', 'CANON'),
                        ('PYTOMOGRAPHY', 'PYTOMOGRAPHY')):
        if name.startswith(key):
            return vendor
    return 'GE' if name == 'GE' else name
