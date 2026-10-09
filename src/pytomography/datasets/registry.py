"""Every dataset the tutorials read: what it is, where it is published, and how to download and check it.

Keys are folders under the data folder (``PYTOMOGRAPHY_DATA``). :func:`pytomography.datasets.fetch` reads this file,
and so do the "Tutorial data" page of the docs and the data badge of each tutorial, so a dataset is added, moved or
re-pinned here and nowhere else. The file has no imports, so the docs can load it without importing PyTomography.

A dataset has:

    title, source, url  what it is, who published it, and its landing page (a DOI when there is one)
    licence, cite       the licence of the data, and what to cite when you use them
    tutorials           the tutorial notebooks that read it
    parts               what to download; every part is pinned by size and checksum
    extras              optional, named groups of parts that ``fetch(name, extras=[...])`` adds
    status, note        "pending" while the data cannot be downloaded yet; ``note`` says why

A part is one of:

    zip        an archive to download and unpack. ``prefix`` keeps only the members under it (and removes it from
               their paths); ``exclude`` drops members matching these patterns; ``unpacked`` is their size on disk.
    zip_range  members of a larger zip on a server, read with HTTP Range requests: ``ranges`` are [first byte,
               end byte, sha256] of the blocks that hold them, and ``index`` is [first byte, sha256] of the zip's
               central directory, which runs to the end of the zip. ``prefix``, ``exclude`` and ``unpacked`` as for zip.
    file       one file, saved as ``to`` (default: its name on the server).
    package    one small file shipped in pytomography/datasets/files, copied to ``to`` (default: the same name).
    idc        one DICOM series from the NCI Imaging Data Commons, downloaded with idc-index into the folder ``to``.
               Pinned by its number of files, bytes and content hash: the sha256 of the sorted sha256 digests of its
               files.

Sizes are in bytes. ``url`` may be replaced by ``urls``, a list of mirrors of the same file tried in order.
"""

# The 2025 PyTomography SPECT tutorial record holds one SPECT.zip. Each SPECT dataset is a contiguous block of it, so
# fetch() downloads only that block and the zip's central directory (169 kB) until the SPECT v2 record has one zip
# per dataset. The blocks leave out the old pickled PSF operators and the StarGuide list-mode files.
_SPECT_2025 = {
    "url": "https://zenodo.org/records/15314460/files/SPECT.zip",
    "size": 1430341130,
    "index": [1430172499, "6f3fb65368580bd49e5a06de8f9072c76eb49c7175cf49ee46131c2fdc8d0b03"],
}
_SPECT_2025_DATASET = {
    "source": "PyTomography SPECT tutorial data (Zenodo)",
    "url": "https://doi.org/10.5281/zenodo.15314460",
    "licence": "CC BY 4.0",
    "cite": "Polson L. PyTomography SPECT Tutorial data. Zenodo (2025). doi:10.5281/zenodo.15314460",
}
# The fitted Ac-225 PSF model that replaced the pickled operator in the 2025 record
_AC225_PSF_MODEL = {"kind": "package", "file": "ac225_psf_model.json", "size": 10092,
                    "sha256": "edb5e15edf5b648d948c7d87ed876fdcfef4bea976649eaa000f337db015a168"}

DATASETS = {
    "SPECT/SIMIND-Jaszak": {
        "title": "SIMIND simulations of a Jaszczak phantom (Lu-177 and the Ac-225 decay chain)",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_siminddata", "t_algorithms", "t_accuracy_known_truth", "t_spect_mc", "t_ac225_simind_recon"],
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/SIMIND-Jaszak/", "exclude": ["*.pkl"],
             "ranges": [[64, 710598502, "64ef103e5701ffb73581d1a90212072204c833cdcfe2d4008109cc761625a510"]],
             "unpacked": 1858656588},
            _AC225_PSF_MODEL,
        ],
    },
    "SPECT/Lu177-NEMA-SymT2": {
        "title": "Lu-177 NEMA IQ phantom on a Siemens Symbia T2, with CT and the scanner's reconstruction",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_dicomdata", "t_algorithms", "t_dualpeak", "t_spect_mc2"],
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/Lu177-NEMA-SymT2/",
             "ranges": [[710628303, 721637143, "c7d9149f2f52f6663fb2edbcc1cbcc16d84c4b914dbdc58e0cf4854b86ec2a98"]],
             "unpacked": 71623548},
        ],
    },
    "SPECT/Ac225-NEMA-SymT2": {
        "title": "Ac-225 NEMA IQ phantom on a Siemens Symbia T2, with CT",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_ac225_dicom_recon"],
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/Ac225-NEMA-SymT2/", "exclude": ["*.pkl"],
             "ranges": [[1404147018, 1430015063, "8eeea5d69419f9fb1d6fbe27009b860deb87f36ee27010c68de023e08cbe1c95"]],
             "unpacked": 88245670},
            _AC225_PSF_MODEL,
        ],
    },
    "SPECT/Lu177-PSMA-GEDisc": {
        "title": "Lu-177-PSMA patient scan over two bed positions on a GE Discovery 670 Pro, with CT and organ"
                 " segmentations",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_dicommultibed", "t_uncertainty_spect", "t_plotting"],
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/Lu177-PSMA-GEDisc/",
             "ranges": [[721637143, 806790948, "6935817f70578bd408e53ad87198604ec4ea221e62b98277546571d34e03c79f"]],
             "unpacked": 209193420},
        ],
    },
    "SPECT/Tc99m-Cardiac": {
        "title": "Tc-99m myocardial perfusion rest study from a Siemens SPECT scanner",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_CardiacReorientation"],
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/Tc99m-Cardiac/",
             "ranges": [[1430044867, 1430172499, "d105bb09df5ffd84c7e7d852d4186499f470304b35573ab39c926c1edc479cd1"]],
             "unpacked": 438908},
        ],
    },
    "SPECT/Tc99m-NEMA-Starguide": {
        "title": "Tc-99m NEMA IQ phantom on a GE StarGuide, with CT and the vendor reconstruction",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_starguide"],
        # The two list-mode files (566 MB) are left out: no tutorial reads them
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/Tc99m-NEMA-Starguide/",
             "exclude": ["NM_files/i187957.NMDC.0", "NM_files/i187970.NMDC.0"],
             "ranges": [[806790948, 831564525, "565458a556c734d440706f4c29fd192fb5c761f9a72f2c7ead471e049abab692"],
                        [1118750540, 1125355907, "17301e9a95dfe578266fd338baa9df21e8db3b6d1ec5b46b287b54ee0749c828"],
                        [1404021187, 1404147018, "ae6a80d7ddf2678937515c3684fcc6adde63eab9ea6a438727d9e7d26ff46b75"]],
             "unpacked": 105203284},
        ],
    },
    "PET/GATE-mMR-Brain": {
        "title": "GATE simulation of an FDG brain phantom on a Siemens Biograph mMR, with a normalisation scan",
        "source": "PyTomography PET tutorial data, with the phantom by Belzunce (Zenodo)",
        "url": "https://doi.org/10.5281/zenodo.8045458",
        "licence": "to be confirmed (the phantom images: CC BY 4.0)",
        "cite": "Belzunce MA. High-Resolution Heterogeneous Digital PET [18F]FDG Brain Phantom based on the BigBrain"
                " Atlas. Zenodo (2018). doi:10.5281/zenodo.8045458",
        "tutorials": ["t_pet_introduction", "t_PETGATE_scat_sino", "t_PETGATE_scat_sinoTOF", "t_PETGATE_scat_lm",
                      "t_PETGATE_scat_lmTOF", "t_PETGATE_DIP"],
        "status": "pending",
        "note": "The GATE simulation is being published on Zenodo; it can be downloaded once that record is out."
                " The phantom's MRI and attenuation map come from Belzunce's record, already pinned in its parts.",
        "parts": [
            {"kind": "file", "url": "https://zenodo.org/records/8045458/files/fdg_pet_phantom_mri.nii.gz",
             "size": 471795384, "md5": "e7e899f16596cb95d69370103fd2ec7d"},
            {"kind": "file", "url": "https://zenodo.org/records/8045458/files/fdg_pet_phantom_umap.nii.gz",
             "size": 261908527, "md5": "a531de147927d3c58b038e08a7820ac8"},
        ],
    },
    "PET/GE-DMI-NEMA": {
        "title": "NEMA IQ phantom list mode from a GE Discovery MI PET/CT, with the scanner's corrections",
        "source": "Georg Schramm (Zenodo), with a sphere mask from PyTomography",
        "url": "https://doi.org/10.5281/zenodo.8404015",
        "licence": "CC BY 4.0",
        "cite": "Schramm G. GE Discovery TOF MI PET NEMA IQ projector benchmark listmode data. Zenodo (2023)."
                " doi:10.5281/zenodo.8404015",
        "tutorials": ["t_GE_HDF5", "t_uncertainty_pet"],
        "parts": [
            {"kind": "zip", "url": "https://zenodo.org/records/8404015/files/dmi_nema_lm.zip", "size": 16234398763,
             "md5": "192ca209456f080ab95d03a1ee41d00c", "unpacked": 16719773369},
            # 3D Slicer mask of the six spheres, made for the PET uncertainty tutorial
            {"kind": "package", "file": "fdg_spheres.seg.nrrd", "size": 11399,
             "sha256": "be239bc852277803d7b8a036c629aab003a033269d6763d4b7957c5c95fdab6d"},
        ],
    },
    "PET/PETSIRD-mIEC": {
        "title": "List mode in the PETSIRD format",
        "source": "PyTomography PET tutorial data",
        "url": "https://github.com/ETSInitiative/PETSIRD",
        "licence": "to be confirmed",
        "tutorials": ["t_PETSIRD"],
        "status": "pending",
        "note": "The 2024 example was in an early PETSIRD format and was never published. The PETSIRD tutorial is"
                " moving to a Siemens Biograph mMR NEMA phantom scan (Zenodo 1304454) converted to PETSIRD 0.9.1; this"
                " entry changes when that file is published.",
        "parts": [],
    },
    "CT/ldct-c145": {
        "title": "Chest CT (case C145) from a GE LightSpeed VCT, with the full-dose projections in DICOM-CT-PD format"
                 " and the scanner's reconstruction",
        "source": "The Cancer Imaging Archive, LDCT-and-Projection-data, served by the NCI Imaging Data Commons",
        "url": "https://doi.org/10.7937/9npb-2637",
        "licence": "CC BY 4.0 and the TCIA data usage policy",
        "cite": "McCollough C, Chen B, Holmes D III, Duan X, Yu Z, Yu L, Leng S, Fletcher J. Low Dose CT Image and"
                " Projection Data (LDCT-and-Projection-data), version 7. The Cancer Imaging Archive (2020)."
                " doi:10.7937/9npb-2637. Supported by NIBIB grants EB017095 and EB017185.",
        "tutorials": ["t_CT_GEN3"],
        "parts": [
            {"kind": "idc", "series": "1.2.840.113713.4.100.1.2.234724616712334404725906500152169",
             "crdc": "0b94e451-0b29-48d8-8abd-c2c78ee5111d", "release": "v24", "to": "full_dose_projections",
             "files": 9014, "size": 1073404126,
             "sha256": "ffaacb3857e978ed9bcf7565136be62587261fe9a14571ddf1c02b3c99865c1c"},
            {"kind": "idc", "series": "1.2.840.113713.4.100.1.2.332029262210606042701376029452792",
             "crdc": "02122def-1663-4507-a99e-e21cd8b784e8", "release": "v24", "to": "full_dose_images",
             "files": 316, "size": 166418304,
             "sha256": "be874455af32354452d41db9eb421ab530fb5b6f1cd910446e1ba757602245c4"},
        ],
    },
    "CT/SophiaBeads-256": {
        "title": "Cone-beam micro-CT of glass beads in a plastic tube (Nikon XT H 225), 255 projections",
        "source": "SophiaBeads Dataset Project, University of Manchester (Zenodo)",
        "url": "https://doi.org/10.5281/zenodo.16474",
        "licence": "CC BY-SA 4.0",
        "cite": "Coban SB, McDonald SA. SophiaBeads Dataset Project. Zenodo (2015). doi:10.5281/zenodo.16474",
        "tutorials": ["t_CT_microct"],
        "parts": [
            {"kind": "zip", "url": "https://zenodo.org/records/16474/files/SophiaBeads_256_averaged.zip",
             "size": 1918034418, "md5": "e1a9d42f29922e168da614178f969d0a", "unpacked": 2049216618},
        ],
    },
}
