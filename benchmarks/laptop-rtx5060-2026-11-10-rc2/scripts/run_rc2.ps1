# PyTomography 4.0.0rc2 on the laptop, overnight, one job at a time:
#   tutorials  every tutorial but the Monte Carlo two, as run_ordered.ps1 ran them on rc1 (9 Nov): one runner call per
#              tutorial, the same order and flags, PYTOMOGRAPHY_OUTPUT shared per desktop round (new, empty folders)
#   mc         t_spect_mc and t_spect_mc2, with the 3 h per-cell limit their rc1 runs had
#   dip8       t_PETGATE_DIP with START_CHANNELS = 8 (as the 10 Nov rc1 test), from an untracked copy
#   bench      the reconstruction benchmarks (feature/benchmarks), with the PET events the tutorials saved
# The pytomo-rc2 environment is a clone of pytomo-rc1 with only PyTomography upgraded. spectarf's ARF cache and CuPy's
# kernel cache go to new, empty folders, as the rc1 run started with them empty. A tutorial with a report already is
# skipped, so the script can be re-run to resume.
param([string]$phases = 'tutorials,mc,dip8,bench')
$envroot = "$env:USERPROFILE\miniforge3\envs\pytomo-rc2"
$py      = "$envroot\python.exe"
$repo    = "$env:USERPROFILE\dev\pt-rc2"
$benchrepo = "$env:USERPROFILE\dev\PyTomography-benchmarks"
$simind  = 'C:\simind\smc_win_intel_64_current'
$base    = 'C:\v4runs\laptop-rc2'
$prefix  = "$simind;$envroot;$envroot\Library\mingw-w64\bin;$envroot\Library\usr\bin;$envroot\Library\bin;$envroot\Scripts"

$plan = [ordered]@{
  spect1 = @{ round = 1; t = 't_siminddata,t_dicomdata,t_algorithms,t_accuracy_known_truth,t_dicommultibed,t_dualpeak,t_uncertainty_spect,t_starguide,t_ac225_simind_recon,t_CardiacReorientation,t_algorithms_measured,t_fbp_spect' }
  spect2 = @{ round = 2; t = 't_arf_psf,t_ac225_dicom_recon,t_ac225_dualpeak,t_cgls,t_kernel_deep_kernel_spect_recon' }
  any    = @{ round = 1; t = 't_examplesystemmatrix,t_plotting' }
  ct     = @{ round = 1; t = 't_CT_GEN3,t_CT_microct,t_fbp_ct,t_pct_abdomen' }
  pet1   = @{ round = 1; t = 't_pet_introduction,t_GE_HDF5,t_uncertainty_pet,t_PETSIRD' }
  pet2   = @{ round = 2; t = 't_PETGATE_scat_sino,t_PETGATE_scat_sinoTOF,t_PETGATE_scat_lm,t_PETGATE_scat_lmTOF,t_PETGATE_DIP' }
  mc     = @{ round = 2; t = 't_spect_mc,t_spect_mc2' }
  dip8   = @{ round = 2; t = 't_PETGATE_DIP_c8' }
}

$env:PATH = "$simind;" + $env:PATH
$env:SMC_DIR = "$simind\smc_dir\"
$env:PYTOMOGRAPHY_DATA = 'C:\pytomography_data'
$env:SPECTARF_CACHE = "$base\cache\spectarf"
$env:CUPY_CACHE_DIR = "$base\cache\cupy"
Remove-Item Env:PYTHONPATH, Env:PYTOMOGRAPHY_OUTPUT, Env:__COMPAT_LAYER -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$base\runs", "$base\cache\spectarf", "$base\cache\cupy" | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
function Say([string]$m) { $m; [IO.File]::AppendAllText("$base\progress.log", "$m`r`n", $utf8) }

function Run-Group([string]$g, [int]$timeout) {
  $item = $plan[$g]
  Say "GROUP $g start $(Get-Date -Format s)"
  foreach ($t in $item.t.Split(',')) {
    $out = "$base\runs\$t"
    if (Test-Path "$out\report.json") { Say "SKIP $t (already has a report)"; continue }
    $env:PYTOMOGRAPHY_OUTPUT = "$base\outputs-r$($item.round)"
    Say "START $t $(Get-Date -Format s)"
    $a = @('-u', "$repo\docs\tools\run_tutorials.py", '--python', $py, '--path-prefix', "`"$prefix`"",
           '--out', $out, '--only', $t, '--no-viewer', '--ram-cap-gb', '25', '--max-kernel-gb', '28', '--min-free-gb', '2',
           '--timeout', "$timeout")
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
           -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.WaitForExit()
    if (-not (Test-Path "$out\report.json")) { Say "RESULT $t NO-REPORT (see $out.log / $out.err)"; continue }
    $r = (Get-Content "$out\report.json" -Raw | ConvertFrom-Json)[0]
    $gpu = $null; if ($r.info -and $null -ne $r.info.peak_gpu_gb) { $gpu = $r.info.peak_gpu_gb }
    Say "RESULT $t $($r.status) $($r.wall_time_s)s ram=$($r.peak_kernel_gb) gpu=$gpu"
  }
  Say "GROUP $g done $(Get-Date -Format s)"
}

foreach ($phase in $phases.Split(',')) {
  switch ($phase) {
    'tutorials' { foreach ($g in 'spect1','spect2','any','ct','pet1','pet2') { Run-Group $g 3600 } }
    'mc'        { Run-Group 'mc' 10800 }
    'dip8'      {
      & $py -I "$base\make_dip_c8.py" "$repo\docs\source\notebooks" | ForEach-Object { Say $_ }
      Run-Group 'dip8' 3600
      Remove-Item "$repo\docs\source\notebooks\t_PETGATE_DIP_c8.ipynb" -ErrorAction SilentlyContinue
    }
    'bench'     {
      $env:PYTOMOGRAPHY_OUTPUT = "$base\outputs-r2"
      $out = "benchmarks\results\$(Get-Date -Format yyyy-MM-dd)-rtx-5060-laptop-gpu-rc2"
      Say "BENCH start $(Get-Date -Format s) -> $out"
      $p = Start-Process -FilePath $py -ArgumentList @('-u', 'benchmarks\run.py', '--min-free-gb', '2', '--out', $out) `
             -WorkingDirectory $benchrepo -NoNewWindow -PassThru `
             -RedirectStandardOutput "$base\bench.log" -RedirectStandardError "$base\bench.err"
      $p.WaitForExit()
      Say "BENCH done $(Get-Date -Format s) exit $($p.ExitCode)"
    }
    default { Say "ERROR unknown phase $phase" }
  }
}
Say "ALL DONE $(Get-Date -Format s)"
