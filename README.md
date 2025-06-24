# LISA Project Science FOM

SNR/Cov calculator for LISA Project Science work. 

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
cd ..
uv pip install -e ./lisa-on-gpu
uv pip install -e ./LISAanalysistools
uv pip install -e ./BBHx
uv pip install -e ./FastEMRIWaveforms
uv run jupyter-lab
```

## Authors

* Michael L. Katz
* Robbie Rosati
