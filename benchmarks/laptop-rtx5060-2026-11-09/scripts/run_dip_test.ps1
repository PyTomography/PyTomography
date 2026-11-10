# Issue #439 test: t_PETGATE_DIP with START_CHANNELS = 8, from an uncommitted copy in the PR's worktree, through the
# same runner, environment and flags as the benchmark (run_ordered.ps1), with the round 2 outputs (cached GATE reads)
# as the official t_PETGATE_DIP run had them.
param([string]$name = 't_PETGATE_DIP_c8')
$envroot = "$env:USERPROFILE\miniforge3\envs\pytomo-rc1"
$repo    = "$env:USERPROFILE\dev\PyTomography-dip"
$simind  = 'C:\simind\smc_win_intel_64_current'
$prefix  = "$simind;$envroot;$envroot\Library\mingw-w64\bin;$envroot\Library\usr\bin;$envroot\Library\bin;$envroot\Scripts"
$out     = "C:\v4runs\laptop\dip_tests\$name"
$env:PATH = "$simind;" + $env:PATH
$env:SMC_DIR = "$simind\smc_dir\"
$env:PYTOMOGRAPHY_DATA = 'C:\pytomography_data'
Remove-Item Env:PYTHONPATH, Env:__COMPAT_LAYER -ErrorAction SilentlyContinue
$env:PYTOMOGRAPHY_OUTPUT = 'C:\v4runs\laptop\outputs-r2'
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
$a = @('-u', "$repo\docs\tools\run_tutorials.py", '--python', "$envroot\python.exe", '--path-prefix', "`"$prefix`"",
       '--out', $out, '--only', $name, '--no-viewer', '--ram-cap-gb', '25', '--max-kernel-gb', '28', '--min-free-gb', '2')
$p = Start-Process -FilePath "$envroot\python.exe" -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
       -RedirectStandardOutput "$out.log" -RedirectStandardError "$out.err"
try { $p.PriorityClass = 'BelowNormal' } catch {}
$p.WaitForExit()
if (Test-Path "$out\report.json") { Get-Content "$out\report.json" -Raw } else { "NO REPORT; see $out.log" }
