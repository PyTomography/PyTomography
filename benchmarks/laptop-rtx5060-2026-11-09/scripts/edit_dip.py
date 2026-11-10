"""Issue #439: one cell near the top of t_PETGATE_DIP sets the DIP network's size, with the GPU memory each size takes.

    python -I edit_dip.py <notebook.ipynb>

Edits only the cells named below, by their ids, and writes the notebook back as nbformat does (indent 1, LF, CRLF in
the working tree as git checks it out), so the outputs and every other cell stay byte for byte as they were.
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text(encoding='utf8')
crlf = '\r\n' in raw
nb = json.loads(raw.replace('\r\n', '\n'))
cells = {c.get('id'): c for c in nb['cells']}


def lines(text: str) -> list:
    out = text.split('\n')
    return [l + '\n' for l in out[:-1]] + ([out[-1]] if out[-1] else [])


def replace(cell_id: str, old: str, new: str):
    src = ''.join(cells[cell_id]['source'])
    assert src.count(old) == 1, (cell_id, old)
    cells[cell_id]['source'] = lines(src.replace(old, new))


# The memory section (markdown)
cells['a88242fd']['source'] = lines(
    "## How much memory this needs\n"
    "\n"
    "The 107 million list-mode events are kept on the GPU, and fitting the network takes most of the rest: with the "
    "network these docs use (`START_CHANNELS = 12`), the tutorial peaks at about 8.0 GB of GPU memory, more than an 8 GB "
    "GPU has. The cell below sets the network's size, and its comments give the GPU memory each size takes: "
    "`START_CHANNELS = 8` fits an 8 GB GPU. On Windows, GPU memory counts as RAM too, so the memory budget below leaves "
    "room for the network (`NETWORK_GB`): the scatter estimate (a 34.6 GB TOF sinogram if held whole) is computed one "
    "group of angles at a time, within what is left. The OSEM reconstruction and the scatter estimate print their "
    "estimates before they run (`estimate_memory`); the network's `NETWORK_GB` come on top. The tutorial peaks at about "
    "22 GB of RAM (measured on Windows), and needs about that whatever `RAM_GB` is, so it does not fit in 16 GB: the "
    "events and their randoms, scatter and sensitivity are held whole, which the budget does not split yet.")

# The settings cell (code): RAM, and the network's size
cells['afd0e064']['source'] = lines(
    "RAM_GB = 25 # memory (RAM) this tutorial may use, in GB; it needs about 22 GB however low this is\n"
    "\n"
    "# The Deep Image Prior's network: a UNet (defined below) whose 5 levels are 1, 2, 4, 8 and 16 times START_CHANNELS\n"
    "# wide. Fitting it takes most of the GPU memory this tutorial needs; the list-mode events, their additive term and\n"
    "# the images keep about 2.5 GB more on the GPU for the whole run. GPU memory in GB, as\n"
    "# torch.cuda.max_memory_allocated counts it, with PyTomography 4.0.0rc1 and PyTorch 2.14:\n"
    "#\n"
    "#   START_CHANNELS   fitting the network   the tutorial's peak\n"
    "#   12               5.5                   8.0   more than an 8 GB GPU has; the images on this page\n"
    "#   10               4.7                   7.2\n"
    "#    8               3.7                   6.2   fits an 8 GB GPU\n"
    "#    6               2.9                   5.4\n"
    "#\n"
    "# PyTorch keeps freed blocks for reuse, so the GPU needs some room above these.\n"
    "START_CHANNELS = 12\n"
    "LBFGS_HISTORY = 10 # past steps L-BFGS keeps, 2 copies of the network's parameters each: 0.24 GB at 12 channels\n"
    "NETWORK_GB = 7 * START_CHANNELS / 12 # what fitting the network takes, in GB, which on Windows counts as RAM too\n"
    "pytomography.set_memory_budget(RAM_GB - NETWORK_GB) # what the scatter estimate and the reconstructions may use")

# DIPPrior: the L-BFGS history as an argument
replace('269c80f4', "        lr = 0.1, # learning rate when fitting\n    ):",
        "        lr = 0.1, # learning rate when fitting\n"
        "        history_size = 10, # L-BFGS: past steps kept, 2 copies of the network's parameters each\n    ):")
replace('269c80f4', "        self.lr = lr\n", "        self.lr = lr\n        self.history_size = history_size\n")
replace('269c80f4', "        # history_size=10 as in the paper (100 kept 200 copies of the network's parameters on the GPU)\n",
        "        # history_size: 10 as in the paper (100 kept 200 copies of the network's parameters on the GPU)\n")
replace('269c80f4', "history_size=10, line_search_fn", "history_size=self.history_size, line_search_fn")

# The network and the prior take the settings
cells['1c6e24ba']['source'] = lines(
    "torch.manual_seed(0) # the network's starting weights: the same network, and the same images, on every run\n"
    "net = UNetCustom([START_CHANNELS, 2*START_CHANNELS, 4*START_CHANNELS, 8*START_CHANNELS, 16*START_CHANNELS]).to(pytomography.device)")
replace('acbe85b5', "    lr=0.01\n", "    lr=0.01,\n    history_size=LBFGS_HISTORY,\n")

out = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
path.write_text(out.replace('\n', '\r\n') if crlf else out, encoding='utf8', newline='')
left = [c.get('id') for c in nb['cells'] if 'start_channels' in ''.join(c['source'])]
print('edited; cells still naming start_channels:', left)
