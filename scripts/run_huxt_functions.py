#Script to run just HUXt
import os
import sys
import yaml
import argparse
import psutil
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import threading
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import datetime

import sunpy
import sunpy.map
from sunpy.coordinates.sun import carrington_rotation_number
from sunpy.coordinates import frames
from scipy.interpolate import RegularGridInterpolator, interp1d

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin

from wsaplus import generate_wsaplus_map
import glob
from astropy.io import fits

from pathlib import Path

def read_huxt_output(outdir, cr_num, tag):
    """
    Read HUXt Earth time series output saved via np.savez.

    Parameters
    ----------
    outdir : str or Path
        Directory containing HUXt output files
    cr_num : int
        Carrington rotation number
    tag : str
        Simulation tag

    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'omni_time'
        - 'omni_speed'
        - 'huxt_time'
        - 'huxt_speed'
    """
    outdir = Path(outdir)

    base = f"HUXt_CR{int(cr_num):03d}_{tag}"

    omni_file = outdir / f"{base}_earth_timeseries_omni.npz"
    #huxt_file = outdir / f"{base}_earth_timeseries.npz"

    if not omni_file.exists():
        raise FileNotFoundError(f"OMNI file not found: {omni_file}")

    # --- Load OMNI comparison ---
    omni = np.load(omni_file)
    omni_time = omni["arr_0"]
    omni_speed = omni["arr_1"]

    return {
        "omni_time": omni_time,
        "omni_speed": omni_speed,
        #"huxt_time": huxt_time,
        #"huxt_speed": huxt_speed,
    }

def run_huxt_sim(theta, HUXT_KWARGS, plot=False, outdir=None, event=None):
    """
    Run a HUXt simulation for a given theta using a fixed solar wind boundary.
    """
    inject_hour, longitude, latitude, width, v = theta
    model = H.HUXt(**HUXT_KWARGS)
    dt_cme = inject_hour * u.hour # Add inject_hour to model.initial_time
 
    cme = H.ConeCME(
        t_launch=dt_cme,
        longitude=longitude * u.deg,
        latitude=latitude * u.deg,
        width=width * u.deg,
        v=v * (u.km / u.s),
        thickness=5 * u.solRad,
        cme_expansion=False,
        cme_fixed_duration=False,
    )
 
    model.solve([cme])
    huxt_ts = HA.get_observer_timeseries(model, observer='Earth')
 
    if plot:
        # Standard HUXt's plot_earth_timeseries(model, plot_omni, save, tag) returns (fig, axs)
        # and has no save_ts/outdir kwargs; capture the figure and save it into the event outdir.
        fig_ts, _ = HA.plot_earth_timeseries(model, plot_omni=True, save=False, tag=event)
        fig_ts.savefig(outdir + event + "_earth_timeseries.png")
        plt.close(fig_ts)
 
        t_interest = 7*u.day
        fig, ax = HA.plot(model, t_interest)
        fig.suptitle(f"HUXt Simulation - {event}", y=0.95, color='k', fontweight='bold')
        plt.savefig(outdir + event + "_huxt_equatorial.png")
        plt.close()
 
    starttime = huxt_ts['time'][0]
    endtime = huxt_ts['time'][len(huxt_ts) - 1]
    huxt_time = huxt_ts['time']; huxt_speed =  huxt_ts['vsw']
 
    return huxt_time, huxt_speed, starttime, endtime
 

def interp_omni_huxt(huxt_time, omni_speed,omni_time):
    huxt_t = Time(huxt_time).unix
    omni_t = Time(omni_time).unix

    valid = np.isfinite(omni_speed)
    omni_t_clean = omni_t[valid]
    omni_v_clean = omni_speed[valid]

    interp_omni = interp1d(
        omni_t_clean,
        omni_v_clean,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"  # or np.nan if you prefer
    )

    omni_speed_interp = interp_omni(huxt_t)
    return omni_speed_interp
