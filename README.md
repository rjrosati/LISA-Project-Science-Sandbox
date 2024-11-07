# NASA LISA FOM

SNR/Cov calculator for NASA LISA work. 

1) To run, clone this repository:

```
git clone https://github.com/mikekatz04/NASA_LISA_FOM.git
cd NASA_LISA_FOM
```

2) Run the install script. You must have Anaconda installed with the base distribution loaded. This will run through `conda`. It will create a new environment and install all required packages/versions (LISA Analysis Tools, BBHx, GBGPU, FEW, Eryn, etc.). The default environment name will be "nasa_fom_env." To adjust that, provide `env_name=` after the install script.

```
bash nasa_fom_install.sh env_name=nasa_fom_env
```

3) Activate the environment, open the notebook, and run.

```
cd notebooks/
conda activate env_name
jupyter lab nasa_lisa_fom.ipynb
```

## Authors

* Michael L. Katz
