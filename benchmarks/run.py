"""Run PyTomography's reconstruction benchmarks: each case in a process of its own, one after another.

    python benchmarks/run.py                          # every case
    python benchmarks/run.py --only spect_lu177,pet_gate_listmode/osem_tof
    python benchmarks/run.py --quick                  # a smoke test: one iteration, no repeats (times mean nothing)
    python benchmarks/table.py benchmarks/results/*   # one table, a column per machine

Each case (benchmarks/cases/<case>.py --recon <recon>) reconstructs one tutorial's data with the tutorial's settings
and records how long each step takes and its peak memory (see benchmarks/cases/_bench.py). Each runs in its own
process, so the memory one case leaves behind (PyTorch's and CuPy's caches, fragmentation) never reaches the next,
and a case that fails or runs out of memory loses only itself. Each case also runs under .github/run_capped.py: it is
stopped if it uses more than --max-gb of RAM (on Windows GPU memory counts too) or if the machine's free memory falls
below --min-free-gb. A case is skipped if the machine has less free RAM than the case needs (CASES below) plus
--min-free-gb, or the GPU less free memory than it needs plus --gpu-margin-gb. The PET cases need up to 18 GB: on a
32 GB machine, which has about 21 GB free, run with --min-free-gb 2 (as the tutorial runner there).

Results go to --out (default benchmarks/results/<date>-<GPU>): one JSON file per case, the output of each, and
summary.json and summary.md. The data comes from pytomography.datasets.fetch(), as in the tutorials: set
PYTOMOGRAPHY_DATA (and PYTOMOGRAPHY_OUTPUT, where the PET cases keep the events they read from the ROOT files the first
time, as the tutorials do) before running. Needs psutil (pip install psutil).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (case, recon, GPU GB, RAM GB): what each case needs, to decide whether it can start. Measured on the RTX 5060 Laptop
# GPU (Windows, 10 Nov 2026, benchmarks/results/2026-11-10-rtx-5060-laptop-gpu): GPU = the case's peak plus 0.5 GB
# for the CUDA context; RAM = its peak committed memory (GPU memory counts on Windows), rounded up, plus about 1 GB.
# The order is the run order, the heaviest PET cases last.
CASES = [
    ('spect_lu177', 'osem', 1.0, 6),
    ('spect_lu177', 'bsrem', 1.1, 7),
    ('spect_ac225_arf', 'osem', 5.7, 11),
    ('ct_helical_fbp', 'fbp', 2.6, 11),
    ('ct_microct', 'fdk', 3.3, 8),
    ('ct_microct', 'sart', 6.6, 11),
    ('pct_abdomen', 'fbp', 1.7, 8),
    ('pct_abdomen', 'sart', 1.7, 8),
    ('pet_gate_listmode', 'osem', 3.4, 12),
    ('pet_gate_listmode', 'osem_tof', 3.6, 13),
    ('pet_gate_sinogram', 'osem', 2.1, 15),
    ('pet_gate_sinogram', 'sss', 2.8, 15),
    ('pet_gate_sinogram', 'osem_tof', 6.1, 18),
]


def gpu_free_gb() -> tuple[str, float] | None:
    """(name, free GB) of the first GPU, from nvidia-smi; None without one."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, free = [s.strip() for s in out[0].rsplit(',', 1)]
    return name, float(free) / 1024


def kill_tree(pid: int) -> None:
    """Stops a process and everything it started."""
    import psutil
    try:
        parent = psutil.Process(pid)
        for p in parent.children(recursive=True) + [parent]:
            try:
                p.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        pass


def free_ram_gb() -> float:
    sys.path.insert(0, str(REPO / '.github'))
    from run_capped import free_gb
    return free_gb()


def selected(only: str | None, skip: str | None) -> list[tuple]:
    def match(case, recon, spec):
        return any(s in (case, f'{case}/{recon}') for s in spec.split(','))
    cases = [c for c in CASES if not only or match(c[0], c[1], only)]
    return [c for c in cases if not skip or not match(c[0], c[1], skip)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='cases to run: case or case/recon, comma-separated (default: all)')
    ap.add_argument('--skip', help='cases to leave out, the same way')
    ap.add_argument('--out', type=Path, help='results folder (default benchmarks/results/<date>-<GPU>)')
    ap.add_argument('--python', default=sys.executable, help='the interpreter with PyTomography (default: this one)')
    ap.add_argument('--quick', action='store_true', help='smoke test: one iteration, no repeats')
    ap.add_argument('--device', choices=['cuda', 'cpu'], help='run on this device (default: PyTomography\'s choice)')
    ap.add_argument('--repeats', type=int, default=3, help='timed runs after the first (default 3)')
    ap.add_argument('--max-seconds', type=float, default=90, help='stop repeating after this long (default 90)')
    ap.add_argument('--ram-gb', type=float, default=25, help="PyTomography's memory budget, the tutorials' RAM_GB (25)")
    ap.add_argument('--max-gb', type=float, default=25, help='stop a case that uses more RAM than this (default 25)')
    ap.add_argument('--min-free-gb', type=float, default=10, help='stop a case if less RAM is free (default 10)')
    ap.add_argument('--gpu-margin-gb', type=float, default=0.5,
                    help='GPU memory to keep free beside a case (default 0.5)')
    ap.add_argument('--timeout', type=float, default=3600, help='stop a case after this many seconds (default 3600)')
    ap.add_argument('--list', action='store_true', help='list the cases and stop')
    args = ap.parse_args()

    cases = selected(args.only, args.skip)
    if args.list or not cases:
        for case, recon, gpu, ram in CASES:
            print(f'{case}/{recon}: GPU about {gpu} GB, RAM about {ram} GB')
        return 0 if cases else 1

    gpu = None if args.device == 'cpu' else gpu_free_gb()
    tag = re.sub(r'[^A-Za-z0-9]+', '-', (gpu[0] if gpu else 'cpu').replace('NVIDIA GeForce ', '')).strip('-').lower()
    out = args.out or HERE / 'results' / f"{datetime.now():%Y-%m-%d}-{tag}{'-quick' if args.quick else ''}"
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(cases)} cases; results in {out}', flush=True)

    flags = ['--repeats', str(args.repeats), '--max-seconds', str(args.max_seconds), '--ram-gb', str(args.ram_gb)]
    flags += ['--quick'] if args.quick else []
    flags += ['--device', args.device] if args.device else []
    below_normal = 0x4000 if os.name == 'nt' else 0          # BELOW_NORMAL_PRIORITY_CLASS: the machine stays usable
    summary = []
    for case, recon, need_gpu, need_ram in cases:
        name = f'{case}/{recon}'
        result_file, log_file = out / f'{case}.{recon}.json', out / f'{case}.{recon}.log'
        row = dict(case=case, recon=recon)
        free = gpu_free_gb() if gpu else None
        if free and free[1] < need_gpu + args.gpu_margin_gb:
            row.update(status='skipped', reason=f'{free[1]:.1f} GB of GPU memory free; needs about '
                                                f'{need_gpu + args.gpu_margin_gb:.1f}')
        elif free_ram_gb() < args.min_free_gb + need_ram:
            row.update(status='skipped', reason=f'{free_ram_gb():.1f} GB of RAM free; needs about {need_ram} GB plus '
                                                f'{args.min_free_gb:g} kept free')
        else:
            result_file.unlink(missing_ok=True)
            command = [args.python, str(REPO / '.github' / 'run_capped.py'), '--max-gb', str(args.max_gb),
                       '--min-free-gb', str(args.min_free_gb), '--', args.python, '-u',
                       str(HERE / 'cases' / f'{case}.py'), '--recon', recon, '--out', str(result_file), *flags]
            print(f'\n== {name}', flush=True)
            t0 = time.time()
            with open(log_file, 'w', encoding='utf-8') as log:
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8', errors='replace', creationflags=below_normal)
                timer = threading.Timer(args.timeout, kill_tree, [proc.pid])
                timer.start()
                try:
                    for line in proc.stdout:
                        log.write(line)
                        if line.startswith('[') or 'run_capped' in line or 'Error' in line:
                            print('  ' + line.rstrip(), flush=True)
                    code = proc.wait()
                finally:
                    timer.cancel()
                    if proc.poll() is None:
                        kill_tree(proc.pid)
            row['wall_s'] = round(time.time() - t0, 1)
            if result_file.exists() and code == 0:
                row.update(status='passed', result=json.loads(result_file.read_text(encoding='utf-8')))
            else:
                tail = log_file.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-3:]
                row.update(status='failed' if row['wall_s'] < args.timeout else 'timed out', reason=' | '.join(tail))
        if row['status'] != 'passed':
            print(f"== {name}: {row['status']}: {row.get('reason', '')}", flush=True)
        summary.append(row)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')

    (out / 'summary.md').write_text(markdown(summary), encoding='utf-8')
    print('\n' + markdown(summary))
    return 0 if all(r['status'] == 'passed' for r in summary) else 1


def markdown(summary: list[dict]) -> str:
    machine = next((r['result']['machine'] for r in summary if r['status'] == 'passed'), {})
    lines = [f"{machine.get('gpu', 'CPU')}, {machine.get('cpu', '')}, {machine.get('ram_gb', '')} GB RAM; "
             f"PyTomography {machine.get('packages', {}).get('pytomography', '?')}, "
             f"PyTorch {machine.get('packages', {}).get('torch', '?')}", '',
             '| Case | Step | First run (s) | Repeats (s, median) | Peak RAM (GB) | Peak GPU (GB) |',
             '|---|---|---|---|---|---|']
    for r in summary:
        if r['status'] != 'passed':
            lines.append(f"| {r['case']}/{r['recon']} | {r['status']}: {r.get('reason', '')[:120]} | | | | |")
            continue
        for s in r['result']['stages']:
            first = s.get('first_s', s.get('s'))
            rep = s.get('repeat_median_s', '')
            gpu = max(s['gpu_torch_gb'], s['gpu_device_gb'])
            lines.append(f"| {r['case']}/{r['recon']} | {s['stage']} | {first:.2f} | {rep} | {s['ram_gb']:.1f} | {gpu:.2f} |")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    sys.exit(main())
