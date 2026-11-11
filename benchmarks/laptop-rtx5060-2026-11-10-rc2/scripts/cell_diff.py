"""Cell-by-cell wall times of one tutorial, rc2 against rc1: python -I cell_diff.py <tutorial> [rc1 notebook]"""
import json, sys, datetime as dt
from pathlib import Path
def cells(nb):
    out = {}
    for i, c in enumerate(json.loads(Path(nb).read_text(encoding='utf8'))['cells']):
        m = c.get('metadata', {}).get('execution', {})
        if c['cell_type'] == 'code' and m.get('iopub.execute_input') and m.get('shell.execute_reply'):
            t = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
            out[i] = (round((t(m['shell.execute_reply']) - t(m['iopub.execute_input'])).total_seconds(), 1),
                      ''.join(c['source']).strip().split('\n')[0][:90])
    return out
name = sys.argv[1]
rc1 = sys.argv[2] if len(sys.argv) > 2 else rf'C:\v4runs\laptop\runs\{name}\{name}.ipynb'
a, b = cells(rf'C:\v4runs\laptop-rc2\runs\{name}\{name}.ipynb'), cells(rc1)
rows = sorted(set(a) | set(b), key=lambda i: -abs(a.get(i, (0,))[0] - b.get(i, (0,))[0]))
print(f'{name}: rc2 {sum(v[0] for v in a.values()):.0f} s, rc1 {sum(v[0] for v in b.values()):.0f} s in timed cells')
for i in rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 6]:
    print(f'  cell {i}: rc2 {a.get(i, (None,))[0]} s, rc1 {b.get(i, (None,))[0]} s  {(a.get(i) or b.get(i))[1]}')
