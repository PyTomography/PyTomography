"""Every tutorial on rc2 against this laptop's rc1 run (9-10 Nov): the combined report and the comparison tables.

    python -I compare_tutorials.py <out folder>

Reads C:\\v4runs\\laptop-rc2\\runs\\<t>\\report.json (rc2) and C:\\v4runs\\laptop\\runs\\<t>\\report.json (rc1; the 8-channel
DIP run from dip_tests). Writes to <out folder>:
  report.json   every rc2 tutorial's report, in run order (the runner's format)
  report.md     the runner's table for them
  cell_timings.json   each rc2 code cell's wall time, from the executed notebooks
  compare.md    the comparison tables (pasted into the README) and the flags
Flags: a tutorial that did not pass, a time that changed by more than 10%, a peak (GPU or RAM) that rose by more than
0.3 GB.
"""
import datetime as dt
import json
import sys
from pathlib import Path

RC2, RC1 = Path(r'C:\v4runs\laptop-rc2\runs'), Path(r'C:\v4runs\laptop\runs')
RC1_DIP8 = Path(r'C:\v4runs\laptop\dip_tests\t_PETGATE_DIP_c8')
GROUPS = [
    ('SPECT', 't_siminddata,t_dicomdata,t_algorithms,t_accuracy_known_truth,t_dicommultibed,t_dualpeak,'
              't_uncertainty_spect,t_starguide,t_ac225_simind_recon,t_CardiacReorientation,t_algorithms_measured,'
              't_fbp_spect,t_arf_psf,t_ac225_dicom_recon,t_ac225_dualpeak,t_cgls,t_kernel_deep_kernel_spect_recon'),
    ('Any modality', 't_examplesystemmatrix,t_plotting'),
    ('CT and proton CT', 't_CT_GEN3,t_CT_microct,t_fbp_ct,t_pct_abdomen'),
    ('PET', 't_pet_introduction,t_GE_HDF5,t_uncertainty_pet,t_PETSIRD,t_PETGATE_scat_sino,t_PETGATE_scat_sinoTOF,'
            't_PETGATE_scat_lm,t_PETGATE_scat_lmTOF,t_PETGATE_DIP'),
    ('Monte Carlo SPECT', 't_spect_mc,t_spect_mc2'),
    ('t_PETGATE_DIP at 8 channels', 't_PETGATE_DIP_c8'),
]


def report(folder: Path) -> dict | None:
    f = folder / 'report.json'
    return json.loads(f.read_text(encoding='utf8'))[0] if f.exists() else None


def gpu(r):
    return (r or {}).get('info', {}).get('peak_gpu_gb')


def ram(r):
    return (r or {}).get('peak_kernel_gb')


def cell_seconds(nb: Path) -> list[dict]:
    rows = []
    for i, cell in enumerate(json.loads(nb.read_text(encoding='utf8'))['cells']):
        meta = cell.get('metadata', {}).get('execution', {})
        start, end = meta.get('iopub.execute_input'), meta.get('shell.execute_reply')
        if cell['cell_type'] == 'code' and start and end:
            t = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))  # noqa: E731
            rows.append({'cell': i, 's': round((t(end) - t(start)).total_seconds(), 1),
                         'code': ''.join(cell['source']).strip().split('\n')[0][:100]})
    return rows


def fmt(v, d=1):
    return '' if v is None else f'{v:.{d}f}'


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    reports, timings, lines, flags = [], {}, [], []
    totals = []
    for group, names in GROUPS:
        lines += [f'### {group}', '',
                  '| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        t2 = t1 = 0.0
        for name in names.split(','):
            r2 = report(RC2 / name)
            r1 = report(RC1_DIP8 if name == 't_PETGATE_DIP_c8' else RC1 / name)
            if r2:
                reports.append(r2)
                nb = RC2 / name / f'{name}.ipynb'
                if nb.exists():
                    timings[name] = cell_seconds(nb)
            s2, s1 = (r2 or {}).get('wall_time_s'), (r1 or {}).get('wall_time_s')
            st2, st1 = (r2 or {}).get('status', 'not run'), (r1 or {}).get('status', 'not run')
            f = []
            if st2 != 'passed':
                f.append(f'**{st2}**' + (' (as on rc1)' if st1 == st2 else f' (rc1: {st1})'))
            change = ''
            if s2 and s1 and st2 == st1 == 'passed':
                pct = 100 * (s2 - s1) / s1
                change = f'{pct:+.0f}%'
                if abs(pct) > 10:
                    f.append(f'time {pct:+.0f}%')
                t2, t1 = t2 + s2, t1 + s1
            for label, a, b in (('GPU', gpu(r2), gpu(r1)), ('RAM', ram(r2), ram(r1))):
                if a is not None and b is not None and round(a - b, 2) > 0.3:
                    f.append(f'{label} +{a - b:.2f} GB')
            if f:
                flags.append((name, '; '.join(f)))
            lines.append(f'| {name} | {st2} | {fmt(s2)} | {fmt(s1)} | {change} | {fmt(gpu(r2), 2)} | {fmt(gpu(r1), 2)} | '
                         f'{fmt(ram(r2))} | {fmt(ram(r1))} | {"; ".join(f)} |')
        if t1:
            lines += ['', f'Passed on both: {t2 / 60:.1f} min on rc2, {t1 / 60:.1f} min on rc1 ({100 * (t2 - t1) / t1:+.0f}%).']
            totals.append((group, t2, t1))
        lines.append('')
    head = ['| Group | rc2 (min) | rc1 (min) | Change |', '|---|---|---|---|']
    head += [f'| {g} | {a / 60:.1f} | {b / 60:.1f} | {100 * (a - b) / b:+.0f}% |' for g, a, b in totals]
    a = sum(t[1] for t in totals if not t[0].startswith('t_PETGATE_DIP'))
    b = sum(t[2] for t in totals if not t[0].startswith('t_PETGATE_DIP'))
    head += [f'| **The 33 tutorials that passed on both** | **{a / 60:.1f}** | **{b / 60:.1f}** | **{100 * (a - b) / b:+.0f}%** |', '']
    flag_lines = ['| Tutorial | Flag |', '|---|---|'] + [f'| {n} | {f} |' for n, f in flags]
    (out / 'compare.md').write_text('\n'.join(['## Totals', ''] + head + ['## Flags', ''] + flag_lines + ['', '## Every tutorial', ''] + lines) + '\n',
                                    encoding='utf8')
    (out / 'report.json').write_text(json.dumps(reports, indent=1), encoding='utf8')
    md = ['| Tutorial | Status | Time (s) | Peak GPU (GB) | Peak RAM (GB) | Error |', '|---|---|---|---|---|---|']
    for r in reports:
        err = (f"cell {r['cell']}: " if r['cell'] is not None else '') + (r['error'] or '')
        info = r.get('info', {})
        md.append(f"| {r['notebook']} | {r['status']} | {r['wall_time_s']} | {info.get('peak_gpu_gb', '')} | "
                  f"{info.get('peak_ram_gb') or r.get('peak_kernel_gb', '')} | {err.replace('|', '/')} |")
    (out / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf8')
    (out / 'cell_timings.json').write_text(json.dumps(timings, indent=1) + '\n', encoding='utf8')
    print(f'{len(reports)} reports, {len(flags)} flagged -> {out}')


main()
