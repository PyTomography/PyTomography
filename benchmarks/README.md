# Reconstruction benchmarks

How long PyTomography's main reconstructions take on a machine, and how much memory they need, measured the same way
on every machine. Each case reconstructs one tutorial's data with that tutorial's settings, so a result reads as, for
example, "OSEM 4 x 8 of the Lu-177 NEMA phantom (t_dicomdata) takes 1.2 s on an RTX 5090".

```bash
python benchmarks/run.py --quick      # smoke test first: one iteration each, no repeats (a few minutes; the times mean nothing)
python benchmarks/run.py              # every case
python benchmarks/run.py --only spect_lu177,pet_gate_listmode/osem_tof
python benchmarks/table.py benchmarks/results/<folder> benchmarks/results/<other folder>   # machines side by side
```

Set `PYTOMOGRAPHY_DATA` first (the cases fetch their data with `pytomography.datasets.fetch()`, as the tutorials
do), and `PYTOMOGRAPHY_OUTPUT` to keep the PET events the cases read from the ROOT files the first time (as the
tutorials do; the first PET case then takes a few minutes longer). The organizer needs `psutil`.

## The cases

| Case | What it measures | Tutorial |
|---|---|---|
| `spect_lu177/osem` | SPECT Lu-177, OSEM 4 x 8, attenuation + PSF, 128^3 | t_dicomdata |
| `spect_lu177/bsrem` | the same data, BSREM 40 x 8 with an RDP prior (CT-based weights) | t_algorithms |
| `spect_ac225_arf/osem` | SPECT Ac-225: ARFs generated for the SY-HE collimator, then OSEM 100 x 3 with the ARF PSF | t_ac225_dicom_recon |
| `ct_helical_fbp/fbp` | helical CT FBP, Hann window, 512 x 512 x 316 | t_fbp_ct |
| `ct_microct/fdk`, `/sart` | cone-beam micro-CT, FDK and OS-SART 3 x 5, 500^3 | t_CT_microct |
| `pct_abdomen/fbp`, `/sart` | proton CT, 188.6 million protons: two-pass FBP, and SART along curved paths | t_pct_abdomen |
| `pet_gate_listmode/osem`, `/osem_tof` | PET list mode (mMR, GATE), OSEM 4 x 14, non-TOF and TOF | t_PETGATE_scat_lm, _lmTOF |
| `pet_gate_sinogram/osem`, `/osem_tof` | PET sinograms, OSEM 2 x 24 non-TOF, and TOF | t_PETGATE_scat_sino, _sinoTOF |
| `pet_gate_sinogram/sss` | PET single scatter simulation, non-TOF | t_PETGATE_scat_sino |

`python benchmarks/run.py --list` gives each case's expected peak memory.

## How it measures

- **One process per case.** `run.py` runs each case script (`cases/<case>.py --recon <recon>`) as a process of its own,
  one after another, so the memory a case leaves behind (PyTorch's and CuPy's caches, fragmentation) goes back to the
  system before the next one starts, each case's peaks are its own, and a case that fails or runs out of memory loses
  only itself. Each case runs under `.github/run_capped.py` (stopped above `--max-gb`, default 25 GB, or below
  `--min-free-gb` free, default 10 GB), at below-normal priority, and is skipped if the machine has too little free
  RAM or GPU memory for it. The PET cases need up to 18 GB of RAM and the micro-CT OS-SART 6.6 GB of GPU memory
  (`--list`): on a 32 GB machine, run with `--min-free-gb 2`.
- **Steps.** Inside a case, each step (loading, building the system matrix, generating ARFs, reconstructing) is timed
  on its own, with the GPU synchronised at both ends. Before each step the caches are emptied (outside the clock).
- **First run and repeats.** A reconstruction runs once, then again up to `--repeats` times (default 3, at most
  `--max-seconds` of repeats, default 90). The first run includes one-off work (compiling kernels, each subset's
  sensitivity image); the repeats show the steady state. The tables give the median of the repeats, and the first run
  in brackets.
- **Memory.** Each step records the process's peak RAM (committed memory on Windows, where GPU memory counts too;
  resident memory on Linux), the peak GPU memory PyTorch allocated, and the peak rise in the GPU's used memory, which
  also counts CuPy, parallelproj and spectarf (on a shared GPU, also other processes).

## Results

`run.py` writes `benchmarks/results/<date>-<GPU>/`: one JSON file per case (every step, the settings, the machine:
GPU, driver, CPU, RAM and package versions), each case's output, and `summary.json` and `summary.md`.
