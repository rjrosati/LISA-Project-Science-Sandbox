from typing import Any
import numpy as np
import pandas as pd
from lisatools.diagnostic import snr as snr_func
from lisatools.diagnostic import plot_covariance_contour, plot_covariance_corner
from lisatools.sensitivity import get_sensitivity, A1TDISens
from lisatools.detector import (
    mrdv1,
    scirdv1,
    Orbits,
    DefaultOrbits,
    LISAModel,
    EqualArmlengthOrbits,
)
import shutil
from lisatools.utils.constants import *
from lisatools.sources import (
    BBHCalculationController,
    GBCalculationController,
    EMRICalculationController,
)
from lisatools.sources.emri import EMRITDIWaveform
from lisatools.sources.bbh import BBHSNRWaveform
from lisatools.sources.gb import GBAETWaveform
from lisatools.stochastic import StochasticContribution, StochasticContributionContainer

import os
from abc import ABC
from typing import Union, Tuple
import matplotlib.pyplot as plt
from scipy import interpolate

from eryn.prior import uniform_dist, ProbDistContainer
from eryn.utils import TransformContainer, PeriodicContainer
import numpy as np
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from lisatools.sensitivity import A1TDISens, E1TDISens
import corner
from gbgpu.gbgpu import GBGPU

Tobs = 0.9 * YRSID_SI
dt = 10.0
Nobs = int(Tobs / dt)
Tobs = Nobs * dt

psd_kwargs = dict(
    stochastic_params=(Tobs,),
)

gb_gen = GBGPU()

priors = {
    "gb": ProbDistContainer(
        {
            0: uniform_dist(1e-24, 1e-21),
            # 1: uniform_dist(0.0001, 0.030),
            1: uniform_dist(1e-19, 1e-14),
            2: uniform_dist(0.0, 2 * np.pi),
            3: uniform_dist(
                -0.999, 0.999
            ),  # should be over cosine of inclination, but this is simpler for examples
            4: uniform_dist(0.0, np.pi),
            # 6: uniform_dist(0.0, 2 * np.pi),
            # 7: uniform_dist(-0.999, 0.999),
        }
    )
}


periodic = PeriodicContainer({"gb": {2: 2 * np.pi, 4: np.pi}})


def log_like_fn(x, gb_gen, data, psd, transform_fn, **kwargs):

    x_in = transform_fn.both_transforms(x)
    ll_out = gb_gen.get_ll(x_in.T, data, psd, **kwargs)

    return ll_out

def run_posterior_estimate_vgbs(vgbs, nsteps=4000, burn=5000, filename_base="vgb_sample_storage"):

    ndims = {"gb": 5}
    branch_names = ["gb"]
    nwalkers = 30
    ntemps = 10


    wave_kwargs = dict(T=Tobs, dt=dt, N=256)
    for j in range(len(vgbs)):
        vgb = vgbs.iloc[j]
        try:
            print(f"Starting {vgb["Name"]}.")
            params = np.array(
                [
                    vgb["Amplitude"],
                    vgb["Frequency"],
                    vgb["FrequencyDerivative"],
                    0.0,
                    vgb["InitialPhase"],
                    vgb["Inclination"],
                    vgb["Polarization"],
                    vgb["EclipticLongitude"],
                    vgb["EclipticLatitude"],
                ]
            )

            fill_dict = {
                "ndim_full": 9,
                "fill_inds": np.array([1, 3, 7, 8]),
                "fill_values": np.array(
                    [vgb["Frequency"], 0.0, vgb["EclipticLongitude"], vgb["EclipticLatitude"]]
                ),
            }

            parameter_transforms = {
                5: np.arccos,
            }

            transform_fn = TransformContainer(
                parameter_transforms=parameter_transforms, fill_dict=fill_dict
            )

            A_inj, E_inj = gb_gen.inject_signal(*params, **wave_kwargs)
            data = [A_inj, E_inj]
            df = 1 / Tobs
            freqs = np.arange(A_inj.shape[0]) * df
            psd = [
                get_sensitivity(freqs, sens_fn=A1TDISens, model=scirdv1, **psd_kwargs),
                get_sensitivity(freqs, sens_fn=E1TDISens, model=scirdv1, **psd_kwargs),
            ]
            gb_gen.d_d = np.sum(
                [data[i].conj() * data[i] / psd[i] * 4 * df for i in range(len(data))]
            )

            try:
                del sampler

            except NameError:
                pass

            fp_out = filename_base + f"_{vgb["Name"][2:-1]}.h5"
            if os.path.exists(fp_out):
                print("Copying old file to backup so this can run. PLEASE BE CAREFUL WITH FILES if they exist already.")
                shutil.copy(fp_out, fp_out[:-3] + "_copy_old.h5")
                os.remove(fp_out)
                
            sampler = EnsembleSampler(
                nwalkers,
                ndims,
                log_like_fn,
                priors,
                branch_names=branch_names,
                args=(gb_gen, data, psd, transform_fn),
                kwargs=wave_kwargs,
                vectorize=True,
                backend=fp_out,
                tempering_kwargs=dict(ntemps=ntemps, Tmax=np.inf),
                periodic=periodic,
            )

            sampling_inj = params[transform_fn.fill_dict["test_inds"]]
            sampling_inj[3] = np.cos(sampling_inj[3])

            ll_start = np.zeros((ntemps * nwalkers,))
            factor = 1e-5
            cov_mat = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

            start_params = np.zeros((ntemps * nwalkers, ndims["gb"]))
            tries = 0
            while np.std(ll_start) < 1.0:
                fix = np.full(ntemps * nwalkers, True)
                while np.any(fix):
                    start_params[fix] = sampling_inj * (1.0 + factor * cov_mat * np.random.randn(ntemps * nwalkers, ndims["gb"])[fix])
                    start_params[:] = periodic.wrap({"gb": start_params[:, None]})["gb"][:, 0]
                    logp = priors["gb"].logpdf(start_params)
                    fix = np.isinf(logp)
                    tries += 1
                    if tries > 1000:
                        breakpoint()
                        raise ValueError("Too many tries to start points.")

                ll_start[:] = sampler.compute_log_like(
                    {"gb": start_params.reshape(ntemps, nwalkers, 1, ndims["gb"])}, logp=logp
                )[0].flatten()
                # print(ll_start, np.std(ll_start))
                factor *= 1.5

            start_state = State({"gb": start_params.reshape(ntemps, nwalkers, 1, ndims["gb"])})
            # state
            nsteps = nsteps
            sampler.run_mcmc(start_state, nsteps, burn=burn, progress=True)

            chain = sampler.get_chain(temp_index=0)["gb"].reshape(-1, ndims["gb"])
            fig = corner.corner(
                chain,
                plot_datapoints=False,
                plot_density=False,
                levels=1.0 - np.exp(-0.5 * np.array([1.0, 2.0, 3.0]) ** 2),
                smooth=0.8,
                hist_kwargs=dict(density=True),
            )
            fig.savefig(f"{vgb["Name"][2:-1]}_corner.png", dpi=150)
            amps = chain[:, 0]
            mean_amp = np.mean(amps)
            amps_norm = amps - mean_amp
            amplitude_std = np.std(amps_norm)
            amplitude_error_perc = np.abs(amplitude_std / mean_amp)
            amplidute_mean_diff_from_inj = np.abs(vgb["Amplitude"] - mean_amp)
            amplidute_mean_diff_from_inj_perc = amplidute_mean_diff_from_inj / vgb["Amplitude"]

            print(
                f"{vgb["Name"]}:\n (Injected val: {vgb["Amplitude"]}), Mean amp: {mean_amp}, diff from inj: {amplidute_mean_diff_from_inj}, percent diff from inj: {amplidute_mean_diff_from_inj_perc},\n std around mean: {amplitude_std}, perc error: {amplitude_error_perc}\n\n"
            )

            del sampler, A_inj, E_inj, fig
            print(f"Finished {vgb["Name"]}.")

        except ValueError:
            print(f"Did not work for {vgb["Name"]}.")

if __name__ == "__main__":
    vgbs = pd.read_csv("vgbs.txt")

    keep = [
        "b'AMCVn'",
        "b'SDSSJ1908'",
        "b'HPLib'",
        "b'V803Cen'",
        "b'J0526+5934'",
        "b'HMCnc'",
    ]

    inds_keep = []
    for i in range(len(vgbs)):
        vgb = vgbs.iloc[i]

        if vgb["Name"] in keep:
            inds_keep.append(i)
    inds_keep = np.asarray(inds_keep)
    vgbs_keep = vgbs.iloc[inds_keep]
    run_posterior_estimate_vgbs(vgbs.iloc[1:])
    breakpoint()