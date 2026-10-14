"""Every dataset the tutorials read: what it is, where it is published, and how to download and check it.

The datasets are in :data:`DATASETS`, keyed by their folders under the data folder (``PYTOMOGRAPHY_DATA``).
:func:`pytomography.datasets.fetch` reads this file, and so does the "Tutorial data" page of the docs, so a dataset is
added, moved or re-pinned here and nowhere else. The file has no imports, so the docs can load it without importing
PyTomography.

A dataset has these fields::

    title, source, url  what it is, who published it, and its landing page (a DOI when there is one)
    licence, cite       the licence of the data, and what to cite when you use them
    tutorials           the tutorial notebooks that read it
    parts               what to download; every part is pinned by size and checksum
    extras              optional, named groups of parts that fetch(name, extras=[...]) adds
    status, note        "pending" while the data cannot be downloaded yet; note says why
    geometry            facts about the data that the files do not record, which the readers use when they read the
                        dataset's files: "gate" gives how the phantom sat in GATE's world (world_axes, world_origin
                        in LPS mm; see pytomography.io.PET.read_gate)
    files               of a pending dataset: the files it will hold, {path: [size, sha256]}, pinned before they
                        can be downloaded; they become parts once they have addresses

A part is one of these kinds::

    zip        an archive to download and unpack. prefix keeps only the members under it (and removes it from
               their paths); exclude drops members matching these patterns; unpacked is their size on disk.
    zip_range  members of a larger zip on a server, read with HTTP Range requests: ranges are [first byte,
               end byte, sha256] of the blocks that hold them, and index is [first byte, sha256] of the zip's
               central directory, which runs to the end of the zip. prefix, exclude and unpacked as for zip.
    file       one file, saved as to (default: its name on the server).
    package    one small file shipped in pytomography/datasets/files, copied to to (default: the same name).
    idc        one DICOM series from the NCI Imaging Data Commons, downloaded with idc-index into the folder to.
               Pinned by its number of files, bytes and content hash: the sha256 of the sorted sha256 digests of
               its files.

Sizes are in bytes. ``url`` may be replaced by ``urls``, a list of mirrors of the same file tried in order.

A part of a record that is not published yet has ``"status": "pending"`` and ``record`` (which record it will be in)
instead of an address: a file part with ``to``, pinned by size and sha256. fetch() downloads nothing for it, and the
tutorials read the copy in the dataset's folder; it gets its ``url`` once the record is out. The regions the tutorials
measure (``*_rois.seg.nrrd``) are such parts, or files of a pending dataset, until their records are published.
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
# Records that are not published yet, which pending parts name: the new version of the 2025 SPECT record, with one
# zip per dataset, and PyTomography's copy of the TCIA case C145 (IDC stays its fallback)
_SPECT_V2 = "PyTomography SPECT tutorial data, version 2 (Zenodo; the new version of record 15314460)"
_CT_C145 = "PyTomography's copy of LDCT-and-Projection-data case C145 (Zenodo)"
# The fitted Ac-225 PSF model that replaced the pickled operator in the 2025 record
_AC225_PSF_MODEL = {"kind": "package", "file": "ac225_psf_model.json", "size": 10092,
                    "sha256": "edb5e15edf5b648d948c7d87ed876fdcfef4bea976649eaa000f337db015a168"}

# The 54 runs of PET/GATE-mMR-BrainInWater, simple_phantom/mMR_voxBrain_withSimplePhantom_{i}.root for i = 1 to 54:
# [size, sha256] of each, as staged for its Zenodo record (10 Oct 2026)
_GATE_BRAININWATER_RUNS = [
    [241443822, "7ba1cea6ba46ab3ae8ff3e91459e3f48b66a8a8e74be1cb8fb6aaa0a61de7360"],  # 1
    [241846022, "cdd4448585746565d6bbd1cf3666eb731c90968655a256389eaf6d7584746fb6"],  # 2
    [242527589, "0012a9be4a94290af212facc8943b9ff055fac0259553fce50b21d77deaa9026"],  # 3
    [242054961, "85fd67d4910115f1460475d9bc606cbce9cfc9ffa122e92296fc4c1bc762b8ce"],  # 4
    [241647158, "467bb2051f84234f85290ae578317909b400ebf31285b4e593ca30be0c3296d3"],  # 5
    [241722456, "c5bb72397d6ff0f2700059813eca60a27fd648d659a248bbc08e3b2854f8933d"],  # 6
    [242053483, "f4c37847575321b357f0172c46470b39ca41a1f97a87b0202e95f84ac1f039d2"],  # 7
    [242624347, "068ea5bd71440dc6ed83aa9c573524c60ab7874e2cd7be396a5a54b59308200c"],  # 8
    [242044582, "52a071c1cacf10990742e290b0c639453d36086f106c45da5ed40e4a3d49ba33"],  # 9
    [241940259, "8e889a09a6df36189abd013457151250f044b97b1324b2eb1a340fae0a62069c"],  # 10
    [242543163, "d03188251e9f8d82aa6b2b92a81bd01c852e8a86fb69b103438271d81dfa81c3"],  # 11
    [241849211, "97bdd3ab25b80f42cf9085f8695b479cc00e3c182d3cbdaac39f3830861ac554"],  # 12
    [242826625, "3b98df6910c9dc2bd3f716c79752a7238e2586fc5b62a40f1cbbef935736e9dd"],  # 13
    [242147837, "8193295587a0a59643020432e5c5f2097ac2605900a658126666b94872ac28ed"],  # 14
    [241657090, "3ea0bc68c33548be1b8bc5d5561b0d9b4b52b47b71c4de13baf352eb06e54778"],  # 15
    [242125643, "2a6769b271f9b9a417856639de506a6deecfbfea8316a4b2d3e8ed698e4853a6"],  # 16
    [242134420, "791c31afc86f508d718f60d02daace988073529c2bfa51fc176df09a07083d96"],  # 17
    [241653132, "3af8c2f4f226c2b6fe599e6b5f84610e1d8c789551a387d58a675ba10ceec6cd"],  # 18
    [242431329, "7824d8fb5cb21bc9dd18a5599c68da65912b1fb619e5269e20d2f1e10e6e4a37"],  # 19
    [242536295, "43d6835107ca127afcc7bfb1c023db2c967bc6db68ace6ee7ab086682182af29"],  # 20
    [242025202, "a96982d3522611a79c01969e60bde814990a830fbf61ab7c249b8183f2b24185"],  # 21
    [241949327, "23a308307cfce1d5817940292afd205070072310d3e3bd8dcc10af3fdd3b9011"],  # 22
    [242053094, "623ddc6e8d3d23bacb04e23fae5ac411b398318959ae074b1afb032f993dc3e0"],  # 23
    [241751624, "b8937cafc5e862b5c179bad1a6189fe45efec934cfcc0c0de8ada2fb02bedde7"],  # 24
    [241745733, "cc1c7561a8ae9354f5a779da8a107664a3237ed5fa34424919ef77e4d7812142"],  # 25
    [242432236, "e29921cc809f1a899f350893a2170763d3dec23d28b3445ca9e0d28f66cf252b"],  # 26
    [242433750, "9a32779ce4d51b0a4d1545b685b2046ae0100a71c7c73c8ffcde5f4dcbbdd51c"],  # 27
    [241842235, "57b67737e8db1f8ecbe2b81a10c23950f3add6714878655e27e95e78cafaf575"],  # 28
    [242026742, "c59f506351c05bf5f17bbfb3ad24b4c06f7584488902ce617bfef27327628a93"],  # 29
    [242609410, "7bd3c020c160b0041541253e269c450cae7b187bc7519a617fbbbce6954f806d"],  # 30
    [242231965, "74f4b12baa568e8b2f4ccd7ba3220571169bc6070439ee6d7a33e494010d8339"],  # 31
    [241937129, "3ffbe2c29acd2d9abfdf12e003ec883aea8385ce7db63bf7e703a2d3bb7c5550"],  # 32
    [242637748, "46c391c3182dff42b7aec485e91faf0cca9321457b0260f7de19cbfd769315e4"],  # 33
    [242236100, "728b45004e00fce78646c33221f18668784c3d171161c497edfe91c316f13e97"],  # 34
    [242432094, "155aa1f04210e41ab84a9ff70114dddcb46cd0ae3f69982b69d43010a833e90d"],  # 35
    [242040074, "4d9d3911d13d626c200cf60af6d451d410f4e247861c60eda0922b6a4e5b9ca4"],  # 36
    [241848121, "97885eca7b537b6f5308b78aaff542907e6bd8b9bfc22f96399355206294be7d"],  # 37
    [241811904, "92163aeded0baaead23ce472cb878d8b14cc45748298042ce85323ddd389c834"],  # 38
    [242051593, "dcbd3f44832503f0fe15e105016f22e1b60340d071792f04124694600e18b474"],  # 39
    [242533105, "5a45390a0f17780eca2c2ef763b2a924e906724d8b81d483086e9014341f294b"],  # 40
    [241754962, "aea9cca9d549d45656ecadec6d3ab0a24aebe5a440817d20290ed7b7c4f41a77"],  # 41
    [242136561, "c666343fa9874c17873f59e786b38ae5c8f208133dc5c8cddb0d316a8e044fc8"],  # 42
    [242048020, "4975f4a4da5f71f6c9266b249d440935e0a23c2ee07f748d38d18b6f9da1cb47"],  # 43
    [242334255, "e0938bc41ba1ec7298b560dbac7f9359e85f83b07b7f1b60c4eae83383ed3bb4"],  # 44
    [241652132, "75dbb879cddf607329950a5442f3cb6b1e9b6e316321ff08778e5b71e37f45e9"],  # 45
    [241654308, "f27379717371df2c91585e53bc9121191b90d14cd6ec356e3520250775ae86ef"],  # 46
    [242130629, "80b5922604b54e91b911d76ca90981ebcb8fb62ba4b01620669080cbf83c5cb0"],  # 47
    [242125968, "c220c5af4eb7b334519739693488712f3cb1af625a05844cd4ade08985d09683"],  # 48
    [242439179, "d6f695c31d7eeddcd1fa26a1b5c64e8983ca3a50f760d90fab410b19fa052f88"],  # 49
    [241941043, "2e1987890b68632de0da0e69ea45af5a1bd165fdd9d1632eb49dfa2a0d792acb"],  # 50
    [242433410, "3e1eafcf29c64db9d49b1b042231c71627f7653fd9449bb787afaa70f6a67e9d"],  # 51
    [242236045, "ec6c42aaf474c452684f4fc6e0bd75e1204d0ae309b67d6c3103e9e3f077928a"],  # 52
    [241949626, "6938bcaaca39d9601b5448fdecb310d6c8dc64ec03596553e86bebf3072803f3"],  # 53
    [242039785, "841b51d6e649aef13e5f3b5b2d214ec2090fd77b4cf87032db9ffd10f295d972"],  # 54
]

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
            # The six spheres of the Jaszczak phantom on SIMIND's 512^3 grid, from which jaszak_rois.seg.nrrd was made
            {"kind": "package", "file": "jaszak_spheres.npz", "size": 136312,
             "sha256": "2c0b2891d92f570e436e1c14ba602bd8029201c5b9a048254513717267c2edc0"},
            # The regions that the accuracy tutorial measures (the six spheres and the background), made by
            # docs/tools/make_roi_masks.py
            {"kind": "file", "status": "pending", "record": _SPECT_V2, "to": "jaszak_rois.seg.nrrd", "size": 51595,
             "sha256": "909f1e31ab1bd99af9809ac2e067a008cd821ffbab37762c2c80928df18f0df9"},
        ],
    },
    "SPECT/Lu177-NEMA-SymT2": {
        "title": "Lu-177 NEMA IQ phantom on a Siemens Symbia T2, with CT and the scanner's reconstruction",
        **_SPECT_2025_DATASET,
        "tutorials": ["t_dicomdata", "t_algorithms", "t_dualpeak", "t_spect_mc2", "t_algorithms_measured",
                      "t_fbp_spect"],
        "parts": [
            {"kind": "zip_range", **_SPECT_2025, "prefix": "SPECT/Lu177-NEMA-SymT2/",
             "ranges": [[710628303, 721637143, "c7d9149f2f52f6663fb2edbcc1cbcc16d84c4b914dbdc58e0cf4854b86ec2a98"]],
             "unpacked": 71623548},
            # The regions that the tutorials measure (the six spheres, the lung insert and the background), made by
            # docs/tools/make_roi_masks.py
            {"kind": "file", "status": "pending", "record": _SPECT_V2, "to": "lu177_nema_symt2_rois.seg.nrrd",
             "size": 246468, "sha256": "be3d1878f5693999cc054d69a508df7f8feeb986d985f20c036173a13bbb4b46"},
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
    # The April 2024 simulation. No tutorial reads it now; it is kept for its normalisation scan (normalization_scan/,
    # 36 ROOT files, 46 GB), from which PET/GATE-mMR-BrainInWater's normalisation weights were computed.
    "PET/GATE-mMR-Brain": {
        "title": "GATE simulation of an FDG brain phantom on a Siemens Biograph mMR, with a normalisation scan",
        "source": "PyTomography PET tutorial data, with the phantom by Belzunce (Zenodo)",
        "url": "https://doi.org/10.5281/zenodo.8045458",
        "licence": "to be confirmed (the phantom images: CC BY 4.0)",
        "cite": "Belzunce MA. High-Resolution Heterogeneous Digital PET [18F]FDG Brain Phantom based on the BigBrain"
                " Atlas. Zenodo (2018). doi:10.5281/zenodo.8045458",
        # GATE's world is the phantom's NIfTI flipped head to foot (GATE's x, y, z are the patient's L, P, I), with
        # GATE's origin at the centre of the phantom's MR (fdg_pet_phantom_mri.nii.gz: its sform puts the centre of its
        # middle voxel at RAS (0.35, -17.65, 22.35) mm): the audit of 11 Oct 2026 (pet_verify_gate*) placed the
        # simulated annihilations on the MR, every axis flip worse
        "geometry": {"gate": {"world_axes": "LPI", "world_origin": [-0.35, 17.65, 22.35]}},
        "tutorials": [],
        "status": "pending",
        "note": "The tutorials now read PET/GATE-mMR-BrainInWater, the October 2024 re-simulation. This dataset is"
                " kept for its normalisation scan (normalization_scan/: 36 ROOT files, 46 GB), from which that"
                " dataset's normalisation weights were computed. It cannot be downloaded yet.",
        "parts": [
            {"kind": "file", "url": "https://zenodo.org/records/8045458/files/fdg_pet_phantom_mri.nii.gz",
             "size": 471795384, "md5": "e7e899f16596cb95d69370103fd2ec7d"},
            {"kind": "file", "url": "https://zenodo.org/records/8045458/files/fdg_pet_phantom_umap.nii.gz",
             "size": 261908527, "md5": "a531de147927d3c58b038e08a7820ac8"},
        ],
    },
    # The October 2024 re-simulation, which the GATE tutorials read. The April 2024 simulation above has a GATE artefact
    # (stripes) from its voxelized attenuating phantom. The normalisation weights come with it, so that no one needs the
    # 46 GB normalisation scan they were computed from (in PET/GATE-mMR-Brain).
    "PET/GATE-mMR-BrainInWater": {
        "title": "GATE simulation of an FDG brain source in a 23 cm water cube on a Siemens Biograph mMR, with"
                 " normalisation weights",
        "source": "PyTomography PET tutorial data, with the brain activity from Belzunce's phantom (Zenodo)",
        "url": "https://doi.org/10.5281/zenodo.8045458",
        "licence": "to be confirmed (the brain phantom: CC BY 4.0)",
        "cite": "Belzunce MA. High-Resolution Heterogeneous Digital PET [18F]FDG Brain Phantom based on the BigBrain"
                " Atlas. Zenodo (2018). doi:10.5281/zenodo.8045458",
        # GATE's world is the phantom's NIfTI flipped head to foot (GATE's x, y, z are the patient's L, P, I), with
        # GATE's origin at the centre of the phantom's MR (fdg_pet_phantom_mri.nii.gz: its sform puts the centre of its
        # middle voxel at RAS (0.35, -17.65, 22.35) mm): the audit of 11 Oct 2026 (pet_verify_gate*) placed the
        # simulated annihilations on the MR, every axis flip worse
        "geometry": {"gate": {"world_axes": "LPI", "world_origin": [-0.35, 17.65, 22.35]}},
        "tutorials": ["t_pet_introduction", "t_PETGATE_scat_sino", "t_PETGATE_scat_sinoTOF", "t_PETGATE_scat_lm",
                      "t_PETGATE_scat_lmTOF", "t_PETGATE_DIP"],
        "status": "pending",
        "note": "The GATE simulation (54 runs of 10 s: 116.9 million coincidences and 14.4 million delays), its"
                " water-cube attenuation map and the scanner's normalisation weights are being published on Zenodo;"
                " they can be downloaded once that record is out, with the grey and white matter that the Deep Image"
                " Prior tutorial measures. The MRI that tutorial uses comes from Belzunce's record.",
        "parts": [],
        "files": {
            # Grey and white matter of the brain phantom, from its MR's intensity classes on the Deep Image Prior
            # tutorial's grid (docs/tools/make_tutorial_regions.py, 13 Oct 2026)
            "gate_braininwater_rois.seg.nrrd":
                [12844, "c702e33efaad4160f08f7bf30ff291394f15a04b9f2f416ecf9a876864fa08ab"],
            **{f"simple_phantom/mMR_voxBrain_withSimplePhantom_{i}.root": run
               for i, run in enumerate(_GATE_BRAININWATER_RUNS, start=1)},
            # The water-cube attenuation map: 344 x 344 x 127 float32 voxels of 1.67475 x 1.67475 x 2.0315 mm in the .v
            # file, with two Interfile headers, .hv (which io.load_image and io.attenuation_map read) and .ahv
            "umap_mMR_brainSimplePhantom.hv":
                [762, "53054ffa66434b06f83c2f787f317a6c0462d3288479d73ee8775e41c24a62ab"],
            "umap_mMR_brainSimplePhantom.ahv":
                [568, "11a44826ffbf0f124b56080dd4737a43f92e1e13a9625031b9188cccfb2f197a"],
            "umap_mMR_brainSimplePhantom.v":
                [60114688, "00110b7f50ea297c59f89b0ba43309b74089d8c03723f74f7f1a755fa2156107"],
            # The scanner, as the simulation's GATE macro describes it
            "mMR_Geometry.mac":
                [2813, "6623f0e2f73c9e7e8a86e1ca2d7ca9eb2e3b231c2148bfd462816ef7d7fdbddb"],
            # The normalisation weight of each of the 411,027,456 crystal pairs (float32, torch.combinations order),
            # from the normalisation scan with gate.get_normalization_weights_cylinder_calibration(...,
            # cylinder_radius=318, include_randoms=False) (io.PET.gate_normalization since PyTomography 4), randoms
            # left out since PyTomography #290. 88,983,552 are NaN: the pairs whose line passes outside the 318 mm
            # calibration shell (io.PET.read_gate sets them to 0).
            "normalization_weights.npy":
                [1644109952, "5f86af901a6c56abad7508f122a621f78497508ba989984b0f8f14a67309e12c"],
        },
        "extras": {
            "mri": {"title": "MRI of the brain phantom, the prior of the Deep Image Prior tutorial", "parts": [
                {"kind": "file", "url": "https://zenodo.org/records/8045458/files/fdg_pet_phantom_mri.nii.gz",
                 "size": 471795384, "md5": "e7e899f16596cb95d69370103fd2ec7d"},
            ]},
        },
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
            # 3D Slicer mask of the six spheres, the lung insert and a background region, made for the PET uncertainty
            # tutorial. Its geometry was corrected for PyTomography 4's upright GE frame (docs/tools/fix_ge_sphere_mask.py,
            # 12 Oct 2026): it was drawn by index on version 3's upside-down grid, with its first two axes swapped. Its
            # pytomography_frame:=scanner field puts it in the GE grid's frame
            {"kind": "package", "file": "fdg_spheres.seg.nrrd", "size": 11411,
             "sha256": "792162c695aa42568ce5f0c3eeb366a156895e621f28f89daa3d9681b39f8873"},
        ],
    },
    # The 60-minute NEMA scan of Zenodo 1304454 converted to PETSIRD 0.9.1 with ETSI's STIR2PETSIRD (October 2026). It
    # replaces PET/PETSIRD-mIEC, a 2024 example in an early PETSIRD format that was never published.
    "PET/PETSIRD-mMR-NEMA": {
        "title": "NEMA IQ phantom on a Siemens Biograph mMR, in the PETSIRD 0.9 format, with its attenuation map",
        "source": "Thomas and Sanderson, University College London (Zenodo), converted to PETSIRD by PyTomography",
        "url": "https://doi.org/10.5281/zenodo.1304454",
        "licence": "CC BY-SA 4.0",
        "cite": "Thomas BA, Sanderson T. NEMA image quality phantom acquisition on the Siemens mMR scanner. Zenodo"
                " (2018). doi:10.5281/zenodo.1304454",
        "tutorials": ["t_PETSIRD"],
        "status": "pending",
        "note": "The scan in PETSIRD 0.9.1 (4.2 GB: 196.6 million prompts and 20.7 million delayed coincidences over"
                " 60 minutes, with the scanner's normalisation) is being published on Zenodo; it can be downloaded once"
                " that record is out, with the regions of the phantom that the tutorial measures. Its attenuation"
                " map, 20170809_NEMA_MUMAP_UCL.v with its .hdr, is in the source record's NEMA_IQ.zip (Zenodo"
                " 1304454).",
        "parts": [],
        "files": {
            # The phantom, its six spheres and its background (also in 10 mm rings round its axis), found from its
            # attenuation map and the tutorial's reconstruction (docs/tools/make_tutorial_regions.py, 13 Oct 2026)
            "petsird_mmr_nema_rois.seg.nrrd":
                [110965, "cd85b63d55651b6ecb6142c211e8fd31616224da85ef88bd566033e672860612"],
        },
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
            # The body, soft tissue and a block of liver, found from the scanner's images on the CT tutorials' grid
            # (docs/tools/make_tutorial_regions.py, 13 Oct 2026)
            {"kind": "file", "status": "pending", "record": _CT_C145, "to": "ldct_c145_rois.seg.nrrd", "size": 1385511,
             "sha256": "fc0553f07b976a5bba4b24d05518422c0aa743e3b2d4197a16353aaf8e68e60d"},
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
"""Every dataset, ``{name: entry}``, where the name is the dataset's folder under the data folder and the entry holds
the fields described at the top of this module."""
