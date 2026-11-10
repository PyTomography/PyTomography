"""One table of benchmark results, a column per machine.

    python benchmarks/table.py benchmarks/results/2026-11-10-rtx-5090 benchmarks/results/2026-11-11-rtx-5060-laptop-gpu

Each cell is the reconstruction step's time: the median of its repeats (the steady state), else its first run; the
first run, which includes one-off work such as compiling kernels and each subset's sensitivity image, is in brackets.
Steps that run once (loading, system matrices, ARFs) follow under each case.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(folder: Path) -> tuple[str, dict]:
    rows = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    machine = next((r['result']['machine'] for r in rows if r['status'] == 'passed'), {})
    label = machine.get('gpu') or f"CPU ({machine.get('cpu', '?')})"
    return label.replace('NVIDIA GeForce ', ''), {(r['case'], r['recon']): r for r in rows}


def cell(row: dict | None, stage: str) -> str:
    if row is None:
        return ''
    if row['status'] != 'passed':
        return row['status']
    s = next((s for s in row['result']['stages'] if s['stage'] == stage), None)
    if s is None:
        return ''
    if 'first_s' in s:
        return f"{s['repeat_median_s']:.2f} ({s['first_s']:.1f})" if 'repeat_median_s' in s else f"{s['first_s']:.2f}"
    return f"{s['s']:.1f}"


def main() -> int:
    machines = [load(Path(f)) for f in sys.argv[1:]]
    if not machines:
        print(__doc__)
        return 1
    keys = list(dict.fromkeys(k for _, rows in machines for k in rows))
    print('| Case | Step | ' + ' | '.join(m for m, _ in machines) + ' |')
    print('|---|---|' + '---|' * len(machines))
    for key in keys:
        rows = [r.get(key) for _, r in machines]
        first = next((r for r in rows if r and r['status'] == 'passed'), None)
        stages = [s['stage'] for s in first['result']['stages']] if first else ['']
        description = first['result']['description'] if first else f'{key[0]}/{key[1]}'
        repeated = [s for s in stages if any('first_s' in s2 for s2 in first['result']['stages'] if s2['stage'] == s)] \
            if first else []
        for stage in repeated + [s for s in stages if s not in repeated]:
            name = f'**{description}**' if stage in repeated[:1] else ''
            print(f"| {name} | {stage} | " + ' | '.join(cell(r, stage) for r in rows) + ' |')
    return 0


if __name__ == '__main__':
    sys.exit(main())
