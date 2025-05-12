# LISA Project Science FOM

SNR/Cov calculator for LISA Project Science work. 

1) To run, clone this repository:

```sh
git clone https://github.com/mikekatz04/LISA-Project-Science-Sandbox.git
cd LISA-Project-Science-Sandbox
```

2) Run the install script. You must have Anaconda installed with the base distribution loaded. This will run through `conda`. It will create a new environment and install all required packages/versions (LISA Analysis Tools, BBHx, GBGPU, FEW, Eryn, etc.). The default environment name will be "fom_env." To adjust that, provide `env_name=` after the install script. (**NOTE**: RIGHT NOW, FEW PIP INSTALL IS NOT WORKING. WILL INSTALL FROM SOURCE inside install script. If you need to reinstall, make sure to delete the FastEMRIWaveforms folder that will download into the main folder of this project. This will be removed in the future. I had it originally automatically deleting the cloned copy after install, but that requires an `rm -rf FastEMRIWaveforms/` running automatically under the hook, SO I DID NOT DO THAT.)

```sh
bash fom_install.sh env_name=fom_env
```



3) Activate the environment, open the notebook, and run.

```sh
cd notebooks/
conda activate $env_name
jupyter lab lisa_proj_sci_fom.ipynb
```

## Authors

* Michael L. Katz
