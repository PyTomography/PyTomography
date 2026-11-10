# Laptop Tutorial Benchmark: the hand-off page's run.ps1, run one tutorial per runner call in Luke's order
# (SPECT, other, CT, PET, then Monte Carlo SPECT last). Same environment, flags and priority as run.ps1.
# PYTOMOGRAPHY_OUTPUT is shared per desktop round (outputs-r1, outputs-r2), as one runner call per round shared
# it on the desktop, so cached steps (e.g. PET normalisation) are reused the same way.
# A tutorial with a report.json already in runs\<name> is skipped, so the script can be re-run to resume.
param([string]$groups = 'spect1,spect2,any,ct,pet1,pet2', [int]$timeout = 3600)  # timeout: the runner's limit per cell, s
$envroot = "$env:USERPROFILE\miniforge3\envs\pytomo-rc1"
$repo    = "$env:USERPROFILE\dev\PyTomography"
$simind  = 'C:\simind\smc_win_intel_64_current'
$base    = 'C:\v4runs\laptop'
$prefix  = "$simind;$envroot;$envroot\Library\mingw-w64\bin;$envroot\Library\usr\bin;$envroot\Library\bin;$envroot\Scripts"

$plan = [ordered]@{
  spect1 = @{ round = 1; t = 't_siminddata,t_dicomdata,t_algorithms,t_accuracy_known_truth,t_dicommultibed,t_dualpeak,t_uncertainty_spect,t_starguide,t_ac225_simind_recon,t_CardiacReorientation,t_algorithms_measured,t_fbp_spect' }
  spect2 = @{ round = 2; t = 't_arf_psf,t_ac225_dicom_recon,t_ac225_dualpeak,t_cgls,t_kernel_deep_kernel_spect_recon' }
  any    = @{ round = 1; t = 't_examplesystemmatrix,t_plotting' }
  ct     = @{ round = 1; t = 't_CT_GEN3,t_CT_microct,t_fbp_ct,t_pct_abdomen' }
  pet1   = @{ round = 1; t = 't_pet_introduction,t_GE_HDF5,t_uncertainty_pet,t_PETSIRD' }
  pet2   = @{ round = 2; t = 't_PETGATE_scat_sino,t_PETGATE_scat_sinoTOF,t_PETGATE_scat_lm,t_PETGATE_scat_lmTOF,t_PETGATE_DIP' }
  mc     = @{ round = 2; t = 't_spect_mc,t_spect_mc2' }
}

$env:PATH = "$simind;" + $env:PATH
$env:SMC_DIR = "$simind\smc_dir\"
$env:PYTOMOGRAPHY_DATA = 'C:\pytomography_data'
Remove-Item Env:PYTHONPATH, Env:PYTOMOGRAPHY_OUTPUT, Env:__COMPAT_LAYER -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$base\runs", "$base\db" | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
function Say([string]$m) { $m; [IO.File]::AppendAllText("$base\progress.log", "$m`r`n", $utf8) }

foreach ($g in $groups.Split(',')) {
  $item = $plan[$g]
  if (-not $item) { Say "ERROR unknown group $g"; continue }
  Say "GROUP $g start $(Get-Date -Format s)"
  foreach ($t in $item.t.Split(',')) {
    $out = "$base\runs\$t"
    if (Test-Path "$out\report.json") { Say "SKIP $t (already has a report)"; continue }
    $env:PYTOMOGRAPHY_OUTPUT = "$base\outputs-r$($item.round)"
    Say "START $t $(Get-Date -Format s)"
    $a = @('-u', "$repo\docs\tools\run_tutorials.py", '--python', "$envroot\python.exe", '--path-prefix', "`"$prefix`"",
           '--out', $out, '--only', $t, '--no-viewer', '--ram-cap-gb', '25', '--max-kernel-gb', '28', '--min-free-gb', '2', '--timeout', "$timeout")
    $p = Start-Process -FilePath "$envroot\python.exe" -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
           -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.WaitForExit()
    if (-not (Test-Path "$out\report.json")) { Say "RESULT $t NO-REPORT (see $out.log / $out.err)"; continue }
    $r = (Get-Content "$out\report.json" -Raw | ConvertFrom-Json)[0]
    $err = $null
    if ($r.error) { $err = ($r.error -split "`r?`n")[0]; if ($err.Length -gt 200) { $err = $err.Substring(0, 200) } }
    $gpu = $null; if ($r.info -and $null -ne $r.info.peak_gpu_gb) { $gpu = $r.info.peak_gpu_gb }
    $doc = [ordered]@{ status = $r.status; wall_time_s = $r.wall_time_s; peak_ram_gb = $r.peak_kernel_gb; peak_gpu_gb = $gpu;
                       round = $item.round; cell = $r.cell; error = $err; date = $r.date; note = $null; group = $g }
    [IO.File]::WriteAllText("$base\db\$t.json", ($doc | ConvertTo-Json -Compress), $utf8)
    Say "RESULT $t $($r.status) $($r.wall_time_s)s ram=$($r.peak_kernel_gb) gpu=$gpu"
  }
  Say "GROUP $g done $(Get-Date -Format s)"
}
Say "ALL DONE $(Get-Date -Format s)"
