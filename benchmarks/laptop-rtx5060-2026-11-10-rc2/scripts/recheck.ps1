# Re-runs one tutorial as a check, with empty spectarf and CuPy caches and an empty output folder, as its first run
# had. For t_ac225_simind_recon, whose ARF generation took 267 s on rc2 against 153 s on rc1 (9 Nov) though spectarf
# is the same: run it once with each environment, one after the other, under the same conditions.
#   powershell -ExecutionPolicy Bypass -File recheck.ps1 -rel rc2 -t t_ac225_simind_recon
param([string]$rel = 'rc2', [string]$t = 't_ac225_simind_recon')
$envroot = "$env:USERPROFILE\miniforge3\envs\pytomo-$rel"
$py      = "$envroot\python.exe"
$repo    = if ($rel -eq 'rc2') { "$env:USERPROFILE\dev\pt-rc2" } else { "$env:USERPROFILE\dev\PyTomography" }
$simind  = 'C:\simind\smc_win_intel_64_current'
$base    = "C:\v4runs\laptop-rc2\recheck\$rel"
$prefix  = "$simind;$envroot;$envroot\Library\mingw-w64\bin;$envroot\Library\usr\bin;$envroot\Library\bin;$envroot\Scripts"
if (Test-Path $base) { Remove-Item -Recurse -Force $base }
New-Item -ItemType Directory -Force "$base\cache\spectarf", "$base\cache\cupy", "$base\outputs" | Out-Null
$env:PATH = "$simind;" + $env:PATH
$env:SMC_DIR = "$simind\smc_dir\"
$env:PYTOMOGRAPHY_DATA = 'C:\pytomography_data'
$env:PYTOMOGRAPHY_OUTPUT = "$base\outputs"
$env:SPECTARF_CACHE = "$base\cache\spectarf"
$env:CUPY_CACHE_DIR = "$base\cache\cupy"
Remove-Item Env:PYTHONPATH, Env:__COMPAT_LAYER -ErrorAction SilentlyContinue
$out = "$base\$t"
$a = @('-u', "$repo\docs\tools\run_tutorials.py", '--python', $py, '--path-prefix', "`"$prefix`"",
       '--out', $out, '--only', $t, '--no-viewer', '--ram-cap-gb', '25', '--max-kernel-gb', '28', '--min-free-gb', '2')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
       -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
try { $p.PriorityClass = 'BelowNormal' } catch {}
$p.WaitForExit()
if (Test-Path "$out\report.json") { $r = (Get-Content "$out\report.json" -Raw | ConvertFrom-Json)[0]; "$rel $t $($r.status) $($r.wall_time_s) s" } else { "$rel $t NO REPORT" }
