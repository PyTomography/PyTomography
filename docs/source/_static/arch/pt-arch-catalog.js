/* Everything the overview diagram's popups list (pt-arch-overview.js). One category per box of the diagram; each item:
   name, tags (modalities or kinds), what it is, its signature, a line of code, its API page and tutorials.
   Links are relative to the site's root. */
(function () {
  const API = "api/pytomography/";
  const T = (name, nb) => [name, `notebooks/${nb}.html`];
  window.PT_CATALOG = {
    files: {
      title: "Data files", kicker: "what the scanners and simulators write",
      blurb: "Every format has a reader. Nothing past the readers ever opens a file.",
      items: [
        { name: "DICOM NM projections", tags: ["SPECT"], what: "GE, Siemens and Mediso scanners. One file holds every view, energy window and time slot.", code: "object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)", api: API + "io/SPECT/dicom/index.html", tut: [T("DICOM Introduction", "t_dicomdata"), T("Multiple bed positions", "t_dicommultibed")] },
        { name: "GE StarGuide", tags: ["SPECT"], what: "Twelve CZT heads, one NM file per head, with body-contour sweeps.", code: "object_meta, proj_meta = dicom.get_starguide_metadata(files_NM)", api: API + "io/SPECT/dicom/index.html", tut: [T("StarGuide Reconstruction", "t_starguide")] },
        { name: "SIMIND interfile", tags: ["SPECT", "simulation"], what: "Monte Carlo SPECT simulations: .h00/.a00 projections, .hct/.ict attenuation maps, .cor orbits.", code: "object_meta, proj_meta = simind.get_metadata(headerfile)", api: API + "io/SPECT/simind/index.html", tut: [T("SIMIND Introduction", "t_siminddata")] },
        { name: "CT slices, RTSTRUCT, NIfTI masks", tags: ["SPECT"], what: "The CT of the study becomes the attenuation map; contours and masks define regions for statistics.", code: "attenuation = SPECTAttenuationTransform(filepath=files_CT)", api: API + "io/SPECT/dicom/index.html", tut: [T("DICOM Introduction", "t_dicomdata")] },
        { name: "GATE (ROOT and .mac)", tags: ["PET", "simulation"], what: "Geometry from the macro file; coincidences, delays and normalisation scans from the ROOT files.", code: "detector_ids = gate.get_detector_ids_from_root(paths, info)", api: API + "io/PET/gate/index.html", tut: [T("GATE list mode with scatter", "t_PETGATE_scat_lm")] },
        { name: "GE Discovery MI (HDF5)", tags: ["PET"], what: "Clinical list mode and the vendor's correction files.", code: "detector_ids = clinical.get_detector_ids_hdf5(listmode_file, scanner_name)", api: API + "io/PET/clinical/index.html", tut: [T("GE Discovery MI", "t_GE_HDF5")] },
        { name: "PETSIRD", tags: ["PET"], what: "The ETSI standard for raw PET data, read with ETSI's own package.", code: "detector_ids, header = petsird.get_detector_ids(file, return_header=True)", api: API + "io/PET/petsird/index.html", tut: [T("PETSIRD", "t_PETSIRD")] },
        { name: "DICOM-CT-PD", tags: ["CT"], what: "Raw projections of 3rd-generation helical scanners, with the flying focal spot and photon counts.", code: "proj, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(paths)", api: API + "io/CT/dicom_ct_pd/index.html", tut: [T("Chest CT from raw projections", "t_CT_GEN3")] },
        { name: "Cone-beam flat panel", tags: ["CT"], what: "Micro-CT projections and angles from the instrument's files, read in the tutorial.", code: "proj_meta = CTConeBeamFlatPanelProjMeta(angles, z_locations, ...)", api: API + "metadata/CT/ct_conebeam_flatpanel_metadata/index.html", tut: [T("Micro-CT", "t_CT_microct")] },
      ],
    },
    readers: {
      title: "Readers", kicker: "pytomography.io",
      blurb: "Each reader turns one format into the standard objects. A new format needs a new reader and nothing else.",
      items: [
        { name: "io.SPECT.dicom", tags: ["SPECT"], what: "Vendor DICOM and StarGuide: metadata, projections, energy-window scatter, attenuation from CT, collimator PSF, multi-bed stitching, masks.", sig: "get_metadata(file, index_peak=0) → (SPECTObjectMeta, SPECTProjMeta)\nget_projections(file, index_peak=None) → tensor\nget_energy_window_scatter_estimate(file, index_peak, index_lower, index_upper) → tensor", api: API + "io/SPECT/dicom/index.html", tut: [T("DICOM Introduction", "t_dicomdata")] },
        { name: "io.SPECT.simind", tags: ["SPECT"], what: "SIMIND interfile: metadata, projections (several windows and organs), attenuation, PSF.", sig: "get_metadata(headerfile) → (SPECTObjectMeta, SPECTProjMeta)\nget_projections(headerfiles, weights=None) → tensor", api: API + "io/SPECT/simind/index.html", tut: [T("SIMIND Introduction", "t_siminddata")] },
        { name: "io.PET.gate", tags: ["PET"], what: "GATE geometry, events, delays, normalisation weights and attenuation maps.", sig: "get_detector_info(path) → info\nget_detector_ids_from_root(paths, info) → detector_ids [N, 2|3]\nget_attenuation_map_nifti(path, object_meta) → μ-map", api: API + "io/PET/gate/index.html", tut: [T("GATE list mode with scatter", "t_PETGATE_scat_lm")] },
        { name: "io.PET.clinical", tags: ["PET"], what: "GE HDF5 list mode, correction weights and the additive term; scanner tables and TOF.", sig: "get_detector_ids_hdf5(listmode_file, scanner_name)\nget_additive_term_hdf5(correction_file)\nget_tof_meta(scanner_name) → PETTOFMeta", api: API + "io/PET/clinical/index.html", tut: [T("GE Discovery MI", "t_GE_HDF5")] },
        { name: "io.PET.petsird", tags: ["PET"], what: "PETSIRD events, the scanner's look-up table, TOF and sensitivity.", sig: "get_detector_ids(file, ...) → detector_ids\nget_scanner_LUT_from_header(header)\nget_TOF_meta_from_header(header)", api: API + "io/PET/petsird/index.html", tut: [T("PETSIRD", "t_PETSIRD")] },
        { name: "io.PET.shared", tags: ["PET"], what: "Format-independent PET: the scanner LUT, and binning between list mode and sinograms.", sig: "listmode_to_sinogram(detector_ids, info) → sinogram\nsinogram_to_listmode(detector_ids, sinogram, info) → values per event", api: API + "io/PET/shared/index.html", tut: [T("GATE sinograms", "t_PETGATE_scat_sino")] },
        { name: "io.CT.dicom_ct_pd", tags: ["CT"], what: "DICOM-CT-PD: the helix, focal spots and photon counts; returns line integrals.", sig: "get_projections_and_metadata_gen3(paths) → (projections, CTGen3ProjMeta)", api: API + "io/CT/dicom_ct_pd/index.html", tut: [T("Chest CT from raw projections", "t_CT_GEN3")] },
        { name: "io.shared.dicom", tags: ["any"], what: "Reconstructed DICOM images (e.g. the scanner's CT) as a volume, with its patient frame.", sig: "open_multifile(files, return_object_meta=False) → volume", api: API + "io/shared/dicom/index.html" },
        { name: "datasets.fetch", tags: ["any"], what: "Downloads, checks and unpacks the tutorial data into PYTOMOGRAPHY_DATA.", code: "folder = pytomography.datasets.fetch(\"SPECT/Lu177-NEMA-SymT2\")" },
      ],
    },
    metadata: {
      title: "Standard objects", kicker: "pytomography.metadata + torch tensors",
      blurb: "Every format ends here: the geometry in metadata objects, the numbers in torch tensors. These are all the system matrix and likelihood ever get.",
      items: [
        { name: "ObjectMeta", tags: ["PET", "CT"], what: "The image grid: voxel size (mm) and shape. Readers add affine_matrix, the patient frame, when they know it.", sig: "ObjectMeta(dr, shape)", api: API + "metadata/metadata/index.html" },
        { name: "SPECTObjectMeta", tags: ["SPECT"], what: "The SPECT image grid, in cm, with the padding the rotations need.", sig: "SPECTObjectMeta(dr, shape)", api: API + "metadata/SPECT/spect_metadata/index.html" },
        { name: "SPECTProjMeta", tags: ["SPECT"], what: "Angles, detector radii (for the PSF) and the projection shape (θ, r, z).", sig: "SPECTProjMeta(projection_shape, dr, angles, radii=None)", api: API + "metadata/SPECT/spect_metadata/index.html" },
        { name: "StarGuideProjMeta", tags: ["SPECT"], what: "Plus each head's transaxial offset and acquisition time.", sig: "StarGuideProjMeta(projection_shape, angles, times, offsets, radii)", api: API + "metadata/SPECT/starguide_metadata/index.html" },
        { name: "SPECTPSFMeta", tags: ["SPECT"], what: "The collimator-detector response: σ as a function of distance, Gaussian or square.", sig: "SPECTPSFMeta(sigma_fit_params, sigma_fit=lambda r, a, b: a * r + b,\n             kernel_dimensions='2D', min_sigmas=3, shape='gaussian')", api: API + "metadata/SPECT/spect_metadata/index.html" },
        { name: "PETLMProjMeta", tags: ["PET"], what: "List mode: detector pairs, the scanner LUT, TOF, per-event weights and the pairs for the sensitivity image.", sig: "PETLMProjMeta(detector_ids, info=None, scanner_LUT=None, tof_meta=None, weights=None, detector_ids_sensitivity=None, weights_sensitivity=None)", api: API + "metadata/PET/petlm_metadata/index.html" },
        { name: "PETSinogramPolygonProjMeta", tags: ["PET"], what: "Sinogram geometry for polygonal scanners.", sig: "PETSinogramPolygonProjMeta(info, tof_meta=None)", api: API + "metadata/PET/pet_sinogram_metadata/index.html" },
        { name: "PETTOFMeta", tags: ["PET"], what: "Time-of-flight bins, range and resolution.", sig: "PETTOFMeta(num_bins, tof_range, fwhm, n_sigmas=3)", api: API + "metadata/PET/pet_tof_metadata/index.html" },
        { name: "CTGen3ProjMeta", tags: ["CT"], what: "Helical geometry: every view's source and detector, the flying focal spot, water attenuation; get_patient_affine gives the patient frame.", sig: "CTGen3ProjMeta(source_phis, source_rhos, source_zs, ..., DSD, shape)", api: API + "metadata/CT/ct_gen3_metadata/index.html" },
        { name: "CTConeBeamFlatPanelProjMeta", tags: ["CT"], what: "Cone-beam geometry with a flat panel.", sig: "CTConeBeamFlatPanelProjMeta(angles, z_locations, detector_radius, beam_radius, shape, dr)", api: API + "metadata/CT/ct_conebeam_flatpanel_metadata/index.html" },
        { name: "Tensors", tags: ["any"], what: "The data itself: projections, events, the additive term, attenuation maps. torch tensors on pytomography.device (the GPU when there is one)." },
      ],
    },
    sysmat: {
      title: "System matrices", kicker: "pytomography.projectors",
      blurb: "H holds all of a scanner's geometry and physics. Likelihoods only call H.forward(f) and H.backward(g), so every algorithm works with every system matrix, yours included.",
      items: [
        { name: "SPECTSystemMatrix", tags: ["SPECT"], what: "Rotates the object to each view and sums along the rays, in pure torch; attenuation and PSF are object transforms applied per view.", sig: "SPECTSystemMatrix(obj2obj_transforms, proj2proj_transforms, object_meta, proj_meta)", code: "SPECTSystemMatrix(obj2obj_transforms=[attenuation, psf], proj2proj_transforms=[],\n                  object_meta=object_meta, proj_meta=proj_meta)", api: API + "projectors/SPECT/dualhead_system_matrix/index.html", tut: [T("DICOM Introduction", "t_dicomdata")] },
        { name: "StarGuideSystemMatrix", tags: ["SPECT"], what: "Twelve heads, each with its offset and acquisition time; the PSF as a grouped convolution.", sig: "StarGuideSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[])", api: API + "projectors/SPECT/starguide_system_matrix/index.html", tut: [T("StarGuide Reconstruction", "t_starguide")] },
        { name: "MonteCarloHybridSPECTSystemMatrix", tags: ["SPECT"], what: "Forward projection by SIMIND Monte Carlo (scatter included), back projection analytic.", sig: "MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, n_events, n_parallel, ...)", api: API + "projectors/SPECT/dualhead_system_matrix/index.html", tut: [T("Hybrid Monte Carlo (SIMIND data)", "t_spect_mc")] },
        { name: "PETLMSystemMatrix", tags: ["PET"], what: "Joseph ray tracing per event, with TOF, on the GPU (parallelproj 2). Attenuation and sensitivity are applied along each line; the sensitivity image is computed once.", sig: "PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[], attenuation_map=None, N_splits=1)", api: API + "projectors/PET/petlm_system_matrix/index.html", tut: [T("GATE list mode with scatter", "t_PETGATE_scat_lm")] },
        { name: "PETSinogramSystemMatrix", tags: ["PET"], what: "The same ray tracing over sinogram bins, with or without TOF.", sig: "PETSinogramSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[], attenuation_map=None, sinogram_sensitivity=None)", api: API + "projectors/PET/pet_sinogram_system_matrix/index.html", tut: [T("GATE sinograms", "t_PETGATE_scat_sino")] },
        { name: "CTGen3SystemMatrix", tags: ["CT"], what: "Helical CT: rays from each focal spot to each detector element. Its analytic inverse is helical FBP (WFBP), with a fused CUDA kernel.", sig: "CTGen3SystemMatrix(object_meta, proj_meta, N_splits=1)", api: API + "projectors/CT/ct_gen3_system_matrix/index.html", tut: [T("Chest CT from raw projections", "t_CT_GEN3")] },
        { name: "CTConeBeamFlatPanelSystemMatrix", tags: ["CT"], what: "Cone beam with a flat panel; its analytic inverse is FDK.", sig: "CTConeBeamFlatPanelSystemMatrix(object_meta, proj_meta, N_splits=1)", api: API + "projectors/CT/ct_conebeam_flatpanel_system_matrix/index.html", tut: [T("Micro-CT", "t_CT_microct")] },
        { name: "KEMSystemMatrix", tags: ["any"], what: "Wraps any system matrix with an anatomical kernel K: H·K, for kernel EM.", sig: "KEMSystemMatrix(system_matrix, kem_transform)", api: API + "projectors/shared/kem_system_matrix/index.html", tut: [T("Algorithms", "t_algorithms")] },
        { name: "ExtendedSystemMatrix", tags: ["any"], what: "Several system matrices as one, e.g. two photopeaks reconstructed together.", sig: "ExtendedSystemMatrix(system_matrices, obj2obj_transforms=None, proj2proj_transforms=None)", api: API + "projectors/system_matrix/index.html", tut: [T("Multi-photopeak reconstruction", "t_dualpeak")] },
        { name: "MotionSystemMatrix", tags: ["any"], what: "Motion correction: one system matrix per frame, each behind a deformation field.", sig: "MotionSystemMatrix(system_matrices, motion_transforms)", api: API + "projectors/shared/motion_correction_system_matrix/index.html" },
        { name: "Your own", tags: ["any"], what: "Subclass SystemMatrix and implement forward, backward and the subset methods. Every likelihood and algorithm then works with it.", code: "class MySystemMatrix(SystemMatrix):\n    def forward(self, object, subset_idx=None): ...\n    def backward(self, proj, subset_idx=None): ...", api: API + "projectors/system_matrix/index.html", tut: [T("Write your own system matrix", "t_examplesystemmatrix")] },
      ],
    },
    obj2obj: {
      title: "Object transforms", kicker: "pytomography.transforms · applied before the projector",
      blurb: "Physics that acts on the image: forward applies it before projecting, backward applies its transpose after back projecting.",
      items: [
        { name: "SPECTAttenuationTransform", tags: ["SPECT"], what: "exp(−∫μ) from each voxel toward the detector, for each view. Built from the CT, a vendor μ-map or a tensor.", sig: "SPECTAttenuationTransform(attenuation_map=None, filepath=None, HU2mu_technique='from_table')", api: API + "transforms/SPECT/attenuation/index.html", tut: [T("DICOM Introduction", "t_dicomdata")] },
        { name: "SPECTPSFTransform", tags: ["SPECT"], what: "The collimator-detector response: a blur that widens with distance from the detector.", sig: "SPECTPSFTransform(psf_meta=None, psf_operator=None)", api: API + "transforms/SPECT/psf/index.html", tut: [T("Ac-225 advanced PSF modelling", "t_ac225_simind_recon")] },
        { name: "GaussianFilter", tags: ["PET", "any"], what: "A Gaussian blur: the PET resolution model, or a filter after reconstruction.", sig: "GaussianFilter(FWHM, n_sigmas=3)", api: API + "transforms/shared/filters/index.html" },
        { name: "KEMTransform", tags: ["any"], what: "The anatomical kernel K of kernel EM, from CT or MR images.", sig: "KEMTransform(support_objects, support_kernels=None, support_kernels_params=None,\n             distance_kernel=None, distance_kernel_params=None, size=5, top_N=None)", api: API + "transforms/shared/kem/index.html", tut: [T("Algorithms", "t_algorithms")] },
        { name: "DVFMotionTransform", tags: ["any"], what: "Deforms the image with a displacement field, for motion correction.", sig: "DVFMotionTransform(dvf_forward, dvf_backward)", api: API + "transforms/shared/motion/index.html" },
        { name: "RotationTransform", tags: ["SPECT"], what: "Rotates the image to each view; the SPECT system matrix uses it inside.", sig: "RotationTransform(mode='bilinear')", api: API + "transforms/shared/spatial/index.html" },
      ],
    },
    projector: {
      title: "Projectors", kicker: "the engine inside each system matrix",
      blurb: "How each system matrix turns an image into projections and back.",
      items: [
        { name: "Rotate and sum", tags: ["SPECT"], what: "Rotate the object so the detector faces +x, apply the object transforms, sum along x. Pure torch, with cached rotation grids." },
        { name: "Head slabs and convolution", tags: ["SPECT"], what: "StarGuide: for each head, the slab of object under it, a grouped 1D convolution for the PSF, and a shift by the head's offset." },
        { name: "Joseph ray tracing with TOF", tags: ["PET"], what: "parallelproj 2 on the GPU, per event or per sinogram bin, with TOF kernels. Events are split into chunks (N_splits) to bound memory." },
        { name: "Helical ray tracing", tags: ["CT"], what: "Rays from each focal spot to each detector element, built on the GPU and traced with parallelproj 2." },
        { name: "Cone-beam ray tracing", tags: ["CT"], what: "Per-view Joseph projection for flat-panel cone beam." },
        { name: "The analytic inverse", tags: ["SPECT", "CT"], what: "Each system matrix can carry its geometry's filtered back projection: parallel-hole FBP (SPECT), helical WFBP (CT, with a fused CUDA kernel) and FDK (cone beam). FilteredBackProjection asks for it.", api: API + "algorithms/fbp/index.html" },
      ],
    },
    proj2proj: {
      title: "Projection transforms", kicker: "pytomography.transforms · applied after the projector",
      blurb: "Effects that act on the projections: forward applies them after projecting, backward before back projecting.",
      items: [
        { name: "CutOffTransform", tags: ["SPECT"], what: "Masks the projections outside a region, e.g. the detector's field of view.", sig: "CutOffTransform(mask)", api: API + "transforms/SPECT/cutoff/index.html" },
        { name: "AdditiveTermTransform", tags: ["SPECT"], what: "Adds a term in forward projection; used by the Monte Carlo hybrid, whose likelihood updates the term.", sig: "AdditiveTermTransform(additive_term)", api: API + "transforms/SPECT/additive_term/index.html", tut: [T("Hybrid Monte Carlo (DICOM data)", "t_spect_mc2")] },
        { name: "Inside the PET system matrices", tags: ["PET"], what: "PET attenuation and detector sensitivity act in projection space too, but the PET system matrices apply them themselves (attenuation_map, weights)." },
      ],
    },
    likelihood: {
      title: "Likelihoods", kicker: "pytomography.likelihoods",
      blurb: "The noise model. It holds the measured data g and the additive term s, and reaches the scanner only through H.forward and H.backward.",
      items: [
        { name: "PoissonLogLikelihood", tags: ["emission"], what: "Counts: ∇L = Hᵀ(g / (Hf + s)) − Hᵀ1. With projections=None it is list mode: each event counts once.", sig: "PoissonLogLikelihood(system_matrix, projections=None, additive_term=None, additive_term_variance_estimate=None)", code: "likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)", api: API + "likelihoods/poisson_log_likelihood/index.html", tut: [T("DICOM Introduction", "t_dicomdata")] },
        { name: "NegativeMSELikelihood", tags: ["any"], what: "Least squares: ∇L = α Hᵀ(g − Hf − s).", sig: "NegativeMSELikelihood(system_matrix, projections=None, additive_term=None, scaling_constant=1.0)", api: API + "likelihoods/mse_objective/index.html", tut: [T("Write your own system matrix", "t_examplesystemmatrix")] },
        { name: "SARTWeightedNegativeMSELikelihood", tags: ["CT"], what: "Least squares weighted by each ray's length, H1; SART builds it for you.", sig: "SARTWeightedNegativeMSELikelihood(system_matrix, projections, additive_term=None)", api: API + "likelihoods/mse_objective/index.html", tut: [T("Chest CT from raw projections", "t_CT_GEN3")] },
        { name: "MonteCarloHybridSPECTPoissonLogLikelihood", tags: ["SPECT"], what: "Poisson, with the Monte Carlo scatter estimate refreshed as the image improves.", api: API + "likelihoods/poisson_log_likelihood/index.html", tut: [T("Hybrid Monte Carlo (SIMIND data)", "t_spect_mc")] },
      ],
    },
    data: {
      title: "Data and additive term", kicker: "g and s, held by the likelihood",
      blurb: "g is what was measured. s is what the system matrix doesn't model but the data contains: scatter and randoms.",
      items: [
        { name: "Projections g", tags: ["SPECT", "PET", "CT"], what: "A tensor of projections (SPECT, sinograms, CT line integrals). For list mode, leave projections out: each event is one count." },
        { name: "Energy-window scatter (DEW, TEW)", tags: ["SPECT"], what: "From the windows next to the photopeak.", code: "scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)", api: API + "utils/scatter/index.html", tut: [T("DICOM Introduction", "t_dicomdata")] },
        { name: "Single scatter simulation", tags: ["PET"], what: "Estimated from a first reconstruction and the attenuation map, scaled to the data.", code: "sinogram_scatter = sss.get_sss_scatter_estimate(object_meta, proj_meta, pet_image, attenuation_image, system_matrix)", api: API + "utils/sss/index.html", tut: [T("GATE list mode with scatter", "t_PETGATE_scat_lm")] },
        { name: "Randoms from delays", tags: ["PET"], what: "Delayed coincidences, binned into a sinogram, smoothed, and read back per event.", code: "sinogram_delays = gate.listmode_to_sinogram(detector_ids_delays, info)", api: API + "io/PET/shared/index.html", tut: [T("GATE list mode with scatter", "t_PETGATE_scat_lm")] },
        { name: "Vendor corrections", tags: ["PET"], what: "GE's correction files give randoms and scatter per event.", code: "additive_term = clinical.get_additive_term_hdf5(correction_file)", api: API + "io/PET/clinical/index.html", tut: [T("GE Discovery MI", "t_GE_HDF5")] },
        { name: "Monte Carlo scatter", tags: ["SPECT"], what: "SIMIND simulates scatter from the current image; refreshed during reconstruction.", api: API + "utils/simind_mc/index.html", tut: [T("Hybrid Monte Carlo (SIMIND data)", "t_spect_mc")] },
      ],
    },
    algorithm: {
      title: "Algorithms", kicker: "pytomography.algorithms",
      blurb: "The update rule. Iterative algorithms ask the likelihood for a gradient and step, f ← f + C(f)·(∇L − ∇V); they never see the data or the scanner.",
      items: [
        { name: "OSEM", tags: ["iterative"], what: "Ordered-subset expectation maximisation, the standard for emission data. C(f) = f / H_mᵀ1.", sig: "OSEM(likelihood, object_initial=None)", code: "recon = OSEM(likelihood)(n_iters=4, n_subsets=8)", api: API + "algorithms/preconditioned_gradient_ascent/index.html", tut: [T("DICOM Introduction", "t_dicomdata"), T("Uncertainty", "t_uncertainty_spect")] },
        { name: "MLEM", tags: ["iterative"], what: "OSEM with one subset.", sig: "MLEM(likelihood, object_initial=None)", code: "recon = MLEM(likelihood)(n_iters=40)", api: API + "algorithms/preconditioned_gradient_ascent/index.html" },
        { name: "OSMAPOSL", tags: ["iterative", "prior"], what: "OSEM with a prior, one step late: C(f) = f / (H_mᵀ1 + ∇V).", sig: "OSMAPOSL(likelihood, object_initial=None, prior=None)", code: "recon = OSMAPOSL(likelihood, prior=prior)(n_iters=4, n_subsets=8)", api: API + "algorithms/preconditioned_gradient_ascent/index.html", tut: [T("Algorithms", "t_algorithms")] },
        { name: "BSREM", tags: ["iterative", "prior"], what: "Block sequential regularised EM: converges with a prior, using a relaxation sequence.", sig: "BSREM(likelihood, object_initial=None, prior=None, relaxation_sequence=lambda _: 1)", code: "recon = BSREM(likelihood, prior=RelativeDifferencePrior(beta=25, gamma=2))(n_iters=4, n_subsets=8)", api: API + "algorithms/preconditioned_gradient_ascent/index.html", tut: [T("Algorithms", "t_algorithms"), T("GE Discovery MI", "t_GE_HDF5")] },
        { name: "KEM", tags: ["iterative"], what: "Kernel EM: reconstructs the coefficients of an anatomical kernel, then returns Kα. Needs a KEMSystemMatrix.", sig: "KEM(likelihood, object_initial=None)", api: API + "algorithms/preconditioned_gradient_ascent/index.html", tut: [T("Algorithms", "t_algorithms")] },
        { name: "RBIEM, RBIMAP", tags: ["iterative"], what: "Rescaled block-iterative EM, and its version with a prior.", sig: "RBIEM(likelihood, prior=None)\nRBIMAP(likelihood, prior=None)", api: API + "algorithms/preconditioned_gradient_ascent/index.html" },
        { name: "SART", tags: ["iterative", "CT"], what: "Simultaneous algebraic reconstruction: weighted least squares, for CT. It builds its own likelihood from H and the data.", sig: "SART(system_matrix, projections, additive_term=None, object_initial=None)", code: "recon = SART(system_matrix, proj)(n_iters=3, n_subsets=40)", api: API + "algorithms/preconditioned_gradient_ascent/index.html", tut: [T("Chest CT from raw projections", "t_CT_GEN3")] },
        { name: "FilteredBackProjection", tags: ["analytic"], what: "Ramp-filtered back projection. Analytic, so no likelihood: it asks the system matrix for its geometry's inverse.", sig: "FilteredBackProjection(projections, system_matrix, filter='hann', **options)", code: "recon = FilteredBackProjection(proj, system_matrix, filter=\"hann\")()", api: API + "algorithms/fbp/index.html", tut: [T("Micro-CT", "t_CT_microct")] },
        { name: "DIPRecon", tags: ["network"], what: "Deep image prior: ADMM with an inner OSEM, and a network you supply as the prior.", sig: "DIPRecon(likelihood, prior_network, rho=3e-3)", api: API + "algorithms/dip_recon/index.html", tut: [T("Deep Image Prior", "t_PETGATE_DIP")] },
        { name: "PGAAMultiBedSPECT", tags: ["SPECT"], what: "One algorithm per bed position, subset by subset, stitched into one image.", sig: "PGAAMultiBedSPECT(files_NM, reconstruction_algorithms)", api: API + "algorithms/preconditioned_gradient_ascent/index.html", tut: [T("Multiple bed positions", "t_dicommultibed")] },
      ],
    },
    prior: {
      title: "Priors", kicker: "pytomography.priors · optional",
      blurb: "A penalty V(f) on the image. Priors plug into the algorithm, never the likelihood.",
      items: [
        { name: "RelativeDifferencePrior", tags: ["edge-preserving"], what: "Penalises relative differences between neighbours; supports uncertainty (second derivatives).", sig: "RelativeDifferencePrior(beta, weight=None, gamma=1, delta=pytomography.delta)", code: "prior = RelativeDifferencePrior(beta=25, gamma=2)", api: API + "priors/nearest_neighbour/index.html", tut: [T("Algorithms", "t_algorithms")] },
        { name: "QuadraticPrior", tags: ["smoothing"], what: "Squared differences between neighbours.", sig: "QuadraticPrior(beta, weight=None, delta=1)", api: API + "priors/nearest_neighbour/index.html" },
        { name: "LogCoshPrior", tags: ["edge-preserving"], what: "Quadratic for small differences, linear for large ones.", sig: "LogCoshPrior(beta, delta=1, weight=None)", api: API + "priors/nearest_neighbour/index.html" },
        { name: "Anatomical neighbour weights", tags: ["CT/MR"], what: "Weight each neighbour by its similarity in a CT or MR image, so edges in the anatomy are kept.", sig: "AnatomyNeighbourWeight(anatomy_image, similarity_function)\nTopNAnatomyNeighbourWeight(anatomy_image, N_neighbours)", api: API + "priors/nearest_neighbour/index.html", tut: [T("Algorithms", "t_algorithms")] },
        { name: "Your own", tags: ["extend"], what: "Subclass NearestNeighbourPrior and give φ and its derivative.", api: API + "priors/prior/index.html" },
      ],
    },
    callback: {
      title: "Callbacks", kicker: "pytomography.callbacks · optional",
      blurb: "Code that runs after every subiteration: record, show or change the current image.",
      items: [
        { name: "Callback", tags: ["base"], what: "Implement run(object, n_iter, n_subset), which returns the (possibly changed) image, and finalize.", api: API + "callbacks/callback/index.html" },
        { name: "DataStorageCallback", tags: ["uncertainty"], what: "Keeps every subiteration's image and forward projection, for compute_uncertainty.", sig: "DataStorageCallback(likelihood, object_initial)", api: API + "callbacks/data_saving/index.html", tut: [T("Uncertainty", "t_uncertainty_spect")] },
      ],
    },
    output: {
      title: "Reconstruction and output", kicker: "a torch tensor on object_meta's grid",
      blurb: "The image is a tensor on the object grid. Saved, every voxel lands in its place in the patient.",
      items: [
        { name: "save_dicom", tags: ["any"], what: "NM (one multi-frame file, the default for SPECT), PT or CT slices. With reference, the series joins the scan's patient and study.", sig: "save_dicom(image, folder, object_meta=None, affine=None, modality=None, reference=None, units=None)", code: "save_dicom(recon, OUTPUT / \"osem\", object_meta, reference=file_NM)", api: API + "io/index.html" },
        { name: "save_nifti", tags: ["any"], what: "Float32 NIfTI with the patient frame.", sig: "save_nifti(image, path, object_meta=None, affine=None, units=None)", api: API + "io/index.html" },
        { name: "Patient frames", tags: ["any"], what: "Where a reader set object_meta.affine_matrix (SPECT DICOM), it is used. Otherwise: gate.get_patient_affine_from_nifti, dicom.get_starguide_patient_affine, or proj_meta.get_patient_affine (CT).", sig: "patient_affine(object_meta=None, affine=None) → 4 × 4" },
        { name: "Uncertainty", tags: ["SPECT", "PET"], what: "Per-region uncertainty from the iterations, for OSEM, OSMAPOSL and BSREM.", code: "algorithm.compute_uncertainty(mask, data_storage_callback)", tut: [T("Uncertainty (SPECT)", "t_uncertainty_spect"), T("Uncertainty (PET)", "t_uncertainty_pet")] },
        { name: "Plots and the 3D viewer", tags: ["any"], what: "Plotting helpers in utils.plot_utils; every tutorial page also opens its results in the 3D viewer.", api: API + "utils/plot_utils/index.html", tut: [T("Plotting", "t_plotting")] },
      ],
    },
  };
})();
