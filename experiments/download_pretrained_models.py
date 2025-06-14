import os
import shutil


print("This script will dowload the trained models, which were used to"
      " compute the errors presented in the paper.")


fnames = ['darcy_flow', 'allen_cahn', 'navier_stokes']

for fid, fname in enumerate(fnames):
    print(f'Downloading {fname} ({fid+1}/{len(fnames)}):')
    url = f"https://zenodo.org/record/6565796/files/{fname}.zip"
    cmd = f"wget --directory-prefix {fname}/ {url}"
    os.system(cmd)
    shutil.unpack_archive(f'{fname}/{fname}.zip', f'{fname}/pretrained_models')
    os.remove(f'{fname}/{fname}.zip')
