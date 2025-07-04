# LISA Project Science FOM

SNR/Cov calculator for LISA Project Science work. 

## Install `uv`
We've set things up to use `uv`, a pretty modern Python environment manager.
You can install it following the directions here: https://docs.astral.sh/uv/getting-started/installation/

## Mac Installation
### Install C dependencies
```sh
brew install openblas gsl
```

Note: if you don't homebrew, edit lines 316-317 of `bbhx_Mmac_setup.py` to point to your OpenBLAS and GSL install paths.
### Installing python dependencies
```sh
git clone --recursive --depth=1 https://github.com/rjrosati/LISA-Project-Science-Sandbox.git
cd LISA-Project-Science-Sandbox
uv sync
cd lisa-on-gpu
uv run --project .. scripts/prebuild.py
cd ../LISAanalysistools
uv run --project .. scripts/prebuild.py
cd ../BBHx
uv run --project .. scripts/prebuild.py
cd ../GBGPU
uv run --project .. scripts/prebuild.py
cd ..
cp bbhx_Mmac_setup.py BBHx/setup.py
uv pip install -e ./lisa-on-gpu
uv pip install -e ./LISAanalysistools
uv pip install -e ./BBHx
uv pip install -e ./GBGPU
```
## Running the notebook
```sh
uv run jupyter-lab
```

## Authors

* Michael L. Katz
* Robbie Rosati (packaging)
