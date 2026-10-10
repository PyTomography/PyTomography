# Tutorial data

Every tutorial reads its data from one folder, named by the `PYTOMOGRAPHY_DATA` environment variable. Put each dataset you need in the subfolder shown below, then point the variable at the folder that contains `SPECT`, `PET` and `CT`.

::::{tab-set}

:::{tab-item} Linux and macOS
```bash
export PYTOMOGRAPHY_DATA=~/pytomography_data
```
:::

:::{tab-item} Windows (PowerShell)
```powershell
setx PYTOMOGRAPHY_DATA "D:\pytomography_data"
```
Open a new terminal afterwards so the variable is picked up.
:::

:::{tab-item} In a notebook
```python
import os
os.environ["PYTOMOGRAPHY_DATA"] = "/path/to/pytomography_data"
```
Run this before the tutorial's data cell.
:::

::::

If the variable is not set, the tutorials look in `pytomography_data` in your home folder. They write their outputs to `pytomography_outputs` in the folder you run them from, or to `PYTOMOGRAPHY_OUTPUT` if you set it, and never into the data folder.

```{note}
On Windows, keep `PYTOMOGRAPHY_OUTPUT` short, for example `D:\pytomography_outputs`. Saved DICOM files are named by their UID, about 64 characters, and Windows limits paths to 260 characters unless long paths are enabled.
```

The layout is:

```text
pytomography_data/
├── SPECT/   SIMIND-Jaszak, Lu177-NEMA-SymT2, Ac225-NEMA-SymT2, Lu177-PSMA-GEDisc, Tc99m-Cardiac, Tc99m-NEMA-Starguide
├── PET/     GATE-mMR-Brain, GE-DMI-NEMA, PETSIRD-mIEC
└── CT/      ldct-c145, SophiaBeads-256
```

You only need the datasets of the tutorials you run.

```{tutorial-datasets}
```
