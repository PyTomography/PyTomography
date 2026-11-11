"""Makes t_PETGATE_DIP_c8.ipynb beside t_PETGATE_DIP.ipynb: the same notebook with START_CHANNELS = 8 (the network size
for an 8 GB GPU, from #442) in place of 12. An untracked copy, for the runner; the tutorial itself is unchanged.

    python -I make_dip_c8.py <notebooks folder>
"""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
nb = json.loads((folder / 't_PETGATE_DIP.ipynb').read_text(encoding='utf8'))
changed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    for i, line in enumerate(cell['source']):
        if line.startswith('START_CHANNELS = 12'):
            cell['source'][i] = line.replace('START_CHANNELS = 12', 'START_CHANNELS = 8', 1)
            changed += 1
if changed != 1:
    sys.exit(f'expected one START_CHANNELS = 12 line, found {changed}')
out = folder / 't_PETGATE_DIP_c8.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf8', newline='\n')
print(f'wrote {out}')
