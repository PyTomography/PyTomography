# Tutorial data

Every tutorial downloads the data it reads in its first code cell, with one line:

```python
from pytomography import datasets

datasets.fetch("SPECT/Lu177-NEMA-SymT2")
```

`fetch()` downloads only that dataset, checks it against the checksums pinned in PyTomography, unpacks it and returns its folder. The next time, it finds the dataset there and returns at once. An interrupted download continues where it stopped, and data already in the folder, from an earlier download or copied there by hand, is checked and used rather than downloaded again. The first download prints the data's licence and what to cite.

## Where the data goes

Into the folder named by the `PYTOMOGRAPHY_DATA` environment variable, or `pytomography_data` in your home folder if the variable is not set. Each dataset is a subfolder, such as `SPECT/Lu177-NEMA-SymT2`.

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

The tutorials write their outputs to `pytomography_outputs` in the folder you run them from, or to `PYTOMOGRAPHY_OUTPUT` if you set it, and never into the data folder.

```{note}
On Windows, keep both folders short, for example `D:\pytomography_data` and `D:\pytomography_outputs`. Some files have long names, such as DICOM files named by their UID, and Windows limits paths to 260 characters unless long paths are enabled. `fetch()` stops before writing anything if a path would be too long.
```

## In Google Colab

```python
!pip install pytomography
from pytomography import datasets
datasets.fetch("SPECT/Lu177-NEMA-SymT2")  # into /root/pytomography_data
```

Colab's disk is wiped when the session ends. To keep the data between sessions, mount your Google Drive and point `PYTOMOGRAPHY_DATA` at it before the first `fetch()`:

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["PYTOMOGRAPHY_DATA"] = "/content/drive/MyDrive/pytomography_data"
```

## From a shell

```bash
python -m pytomography.datasets list                          # every dataset, its size and licence, and whether you have it
python -m pytomography.datasets fetch --tutorial t_dicomdata  # every dataset one tutorial reads
python -m pytomography.datasets verify --hash                 # check every file you downloaded against its checksum
```

In Python, `datasets.available()` gives the same table, and `datasets.info("CT/ldct-c145")` says what a dataset is, where it comes from and how to cite it.

## The datasets

```{tutorial-datasets}
```
