import papermill as pm
import os

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

tutorials = [
    # 't_siminddata.ipynb',
    # 't_dicomdata.ipynb',
    # 't_dicommultibed.ipynb',
    # 't_uncertainty_spect.ipynb',
    # 't_PETGATE_scat_sino.ipynb',
    # 't_PETGATE_scat_sinoTOF.ipynb',
    # 't_PETGATE_scat_lm.ipynb',
    # 't_PETGATE_scat_lmTOF.ipynb',
    # 't_PETGATE_DIP.ipynb',
    't_GE_HDF5.ipynb',
    't_CT_GEN3.ipynb',
    't_examplesystemmatrix.ipynb'
]

for tutorial in tutorials:
    print(f"Running {tutorial}...")
    try:
        pm.execute_notebook(tutorial, tutorial)
        print(f"{tutorial} ran successfully.")
    except pm.PapermillExecutionError as e:
        print(f"{tutorial} failed")