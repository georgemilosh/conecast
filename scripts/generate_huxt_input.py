"""Prepare HUXt event inputs from GONG magnetograms and WSA+ maps.

This script is the background-preparation stage for the GP surrogate workflow.
For each event listed in `data_dir/events.csv` it:

1. downloads or reuses GONG synoptic magnetograms around the CME onset,
2. selects the closest map,
3. runs WSA+ to estimate a longitude-latitude solar-wind speed map,
4. samples that map along the sub-Earth track for the Carrington rotation,
5. writes the 1-degree HUXt boundary file `v_boundary_<event>.npz`, and
6. writes `event_config.yaml` with the seed Cone-CME parameters.

Main outputs are written under the per-event directory: `v_boundary_*.npz`,
`wsaplus_speed_map_*.npz`, diagnostic PNGs, and `event_config.yaml`.
"""

import argparse
import csv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUNPY_CONFIG_DIR = CACHE_DIR / "sunpy"
MPL_CONFIG_DIR = CACHE_DIR / "matplotlib"
SUNPY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SUNPY_CONFIGDIR", str(SUNPY_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import numpy as np
import time
from multiprocessing import Pool
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import datetime
import sunpy
import sunpy.map
from sunpy.coordinates.sun import carrington_rotation_number
from sunpy.coordinates import frames
from scipy.interpolate import RegularGridInterpolator

import huxt.huxt as H
import huxt.huxt_analysis as HA
import huxt.huxt_inputs as Hin

from wsaplus import generate_wsaplus_map
from sunpy.net import Fido, attrs as a

import glob
from astropy.io import fits
import yaml

import run_huxt_functions as rhf

DEFAULT_EVENTS_FILE = BASE_DIR / "data_dir" / "events.csv"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "data_dir" / "sw"
REQUIRED_EVENT_COLUMNS = {
    "event",
    "cme_onset",
    "cme_0p1_au",
    "longitude",
    "latitude",
    "width",
    "speed",
}


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_events(events_file, requested_events=None):
    """Load CME seed parameters from a CSV file."""
    events_file = Path(events_file)
    if not events_file.exists():
        raise FileNotFoundError(f"Missing events CSV: {events_file}")

    with events_file.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_EVENT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{events_file} is missing required columns: {missing_text}")
        rows = list(reader)

    requested = None if not requested_events or requested_events == ["all"] else set(requested_events)
    events = []
    for row in rows:
        event_name = row["event"].strip()
        if requested is not None and event_name not in requested:
            continue
        if "enabled" in row and row["enabled"].strip() and not truthy(row["enabled"]):
            continue
        events.append(
            {
                "event": event_name,
                "cme_onset": row["cme_onset"].strip(),
                "cme_0p1_au": row["cme_0p1_au"].strip(),
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
                "width": float(row["width"]),
                "speed": float(row["speed"]),
            }
        )

    if requested is not None:
        found = {event["event"] for event in events}
        missing_requested = requested - found
        if missing_requested:
            missing_text = ", ".join(sorted(missing_requested))
            raise ValueError(f"Requested events not found or disabled in {events_file}: {missing_text}")
    return events


def download_gong_mag(t_start_mag, t_end_mag, outdir=None):
    """Download GONG synoptic map for a given range of dates
    Input:
    - t_start_mag: start time (astropy Time object)
    - t_end_mag: end time (astropy Time object)
    - outdir: directory to save the downloaded files (default: "gong_mag")
    Output:
    - files: list of downloaded file paths
    """
    
    # Search for GONG synoptic maps in the given time range
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    res = Fido.search(a.Time(t_start_mag, t_end_mag),
                      a.Instrument("GONG"))

    # Check if files already exist
    existing_files = sorted(outdir.glob("*.fits"))

    if len(existing_files) == 0:
        print("Directory empty. Downloading files...")
        files = Fido.fetch(res, path=str(outdir / "{file}"))
        print("Downloaded files:")
        print(files)

    else:
        print("Files already exist. Skipping download.")
        files = existing_files

    for gzfile in outdir.glob("*.fits.gz"):
        fitsfile = gzfile.with_suffix("")
        with fits.open(gzfile) as h:
            h.writeto(fitsfile, overwrite=True)
        gzfile.unlink()
        print("Converted:", gzfile, "→", fitsfile)

    return files

def find_closest_map(target_time, map_dir=None):
    """
    Find the closest map to the target time which is the CME onset time.
    Input:
    - target_time: astropy Time object
    - map_dir: directory containing the maps (default: "gong_mag")
    Output:
    - maps: sunpy map closest to the target time
    - obs_time: observation time of the closest map
    - dt: time difference between target_time and obs_time    
    """

    map_dir = Path(map_dir)
    gong_files = sorted(map_dir.glob("*.fits"))
    print(f"Found {len(gong_files)} GONG files")
    if not gong_files:
        raise FileNotFoundError(f"No GONG .fits files found in {map_dir}")

    maps = []
    obs_times = []

    for f in gong_files:
        try:
            m = sunpy.map.Map(f)
            maps.append(m)
            obs_times.append(m.date)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    obs_times = Time(obs_times)

    dt = np.abs(obs_times - target_time)
    idx = np.argmin(dt)
    
    return maps[idx], obs_times[idx], dt[idx]

def run_wsaplus(filepath, outdir=None, checkpoint_path=None, event=None):
    """
    Run WSA+ to generate a speed map for the given GONG magnetogram file.
    Inputs:
    - filepath: path to the GONG magnetogram file
    - outdir: directory to save the output speed map and plots
    - checkpoint_path: path to the WSA+ checkpoint file
    - event: event name for labeling outputs
    Output:
    - res: WSA+ result object containing the speed map and grid information
    """
    print("Generating WSA+ map for file:", filepath)
    outdir = Path(outdir)
    checkpoint_path = Path(checkpoint_path)
    res = generate_wsaplus_map(filepath, mag_type="GONG", checkpoint_path=str(checkpoint_path))
    np.savez(outdir / f"wsaplus_speed_map_{event}.npz", speed_map=res)
    #Plotting WSA+ map
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(res.phi_grid_deg, res.theta_grid_deg, res.speed_kms, shading='auto', cmap='viridis')
    plt.colorbar(label="v [km/s]")
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title(event)
    plt.savefig(outdir / f"wsaplus_speed_map_{event}.png")
    plt.close()

    return res

def compute_subearth_track(cr_num):
    """
    Compute the sub-Earth track for a given Carrington rotation number.
    Input:
    - cr_num: Carrington rotation number
    Output:
    - SBElon: array of sub-Earth longitudes
    - SBElat: array of sub-Earth latitudes
    """
    t_start = sunpy.coordinates.sun.carrington_rotation_time(cr_num)
    t_end = sunpy.coordinates.sun.carrington_rotation_time(cr_num + 1)

    dt = t_end - t_start
    n_hr = int(dt.value * 24)

    print ('Computing sub-Earth track for CR', cr_num, 'from', t_start, 'to', t_end, 'with', n_hr, 'time steps')

    obs_time = t_start + dt*np.linspace(1e-6, 1-1e-6, n_hr, endpoint=False)

    SBElat = np.zeros(n_hr)
    SBElon = np.zeros(n_hr)

    for i, t in enumerate(obs_time):
        coord = sunpy.coordinates.ephemeris.get_earth(time=t).transform_to(
            frames.HeliographicCarrington(observer='earth')
        )

        SBElat[i] = coord.lat.value
        SBElon[i] = coord.lon.value

    return SBElon, SBElat

def sample_interpolated(speed_map, lon_vals, lat_vals, SBElon, SBElat):
    """
    Sample the speed map at the sub-Earth track using interpolation.
    Input:
    - speed_map: 2D array of speed values on the grid
    - lon_vals: 1D array of longitude values corresponding to the grid
    - lat_vals: 1D array of latitude values corresponding to the grid
    - SBElon: array of sub-Earth longitudes
    - SBElat: array of sub-Earth latitudes
    Output:
    - subearth_speed: array of interpolated speed values at the sub-Earth track"""
    interp_speed = RegularGridInterpolator(
        (lon_vals, lat_vals),
        speed_map,
        bounds_error=False,
        fill_value=np.nan
    )
    subearth_speed = interp_speed(np.column_stack([SBElon, SBElat]))
    return subearth_speed

def sample_nearest(speed_map, lon_vals, lat_vals, SBElon, SBElat):
    """
    Sample the speed map at the sub-Earth track using nearest grid cell values.
    Input:
    - speed_map: 2D array of speed values on the grid
    - lon_vals: 1D array of longitude values corresponding to the grid
    - lat_vals: 1D array of latitude values corresponding to the grid
    - SBElon: array of sub-Earth longitudes
    - SBElat: array of sub-Earth latitudes
    Output:
    - lon_nearest: array of longitudes of the nearest grid cells
    - lat_nearest: array of latitudes of the nearest grid cells
    - speed_nearest: array of speed values at the nearest grid cells"""

    speed_nearest = []
    lon_nearest = []
    lat_nearest = []

    for lon, lat in zip(SBElon, SBElat):

        i_lon = np.argmin(np.abs(lon_vals - lon))
        i_lat = np.argmin(np.abs(lat_vals - lat))

        speed_nearest.append(speed_map[i_lon, i_lat])
        lon_nearest.append(lon_vals[i_lon])
        lat_nearest.append(lat_vals[i_lat])
        #print(f"Sub-Earth point: ({lon:.2f}, {lat:.2f}) → Nearest grid point: ({lon_vals[i_lon]:.2f}, {lat_vals[i_lat]:.2f}), Speed: {speed_map[i_lon, i_lat]:.1f} km/s, i_lon: {i_lon}, i_lat: {i_lat}, shape(speed_map): {len(speed_nearest)}")

    return lon_nearest, lat_nearest, np.array(speed_nearest)

def map_input_huxt(res, outdir=None, event=None, cr_num=None):
    """
    Map the WSA+ speed map to the sub-Earth track and prepare the input for HUXt.
    Input:
    - res: WSA+ result object containing the speed map and grid information
    - outdir: directory to save the output plots
    - event: event name for labeling outputs
    - cr_num: Carrington rotation number (for computing sub-Earth track)
    Output:
    - lon_grid_360: array of longitudes from 1 to 360 degrees
    - speed_nearest_360: array of speed values at the nearest grid cells along the sub-Earth track, interpolated to a 1-degree grid
    - speed_interp_360: array of speed values at the sub-Earth track obtained from interpolation, interpolated to a 1-degree grid
     - Plots comparing the nearest grid cell sampling and interpolation results along the sub-Earth track, and the sub-Earth track on the WSA+ map
    """
    outdir = Path(outdir)
    speed_wsaplus = res.speed_kms
    lon_vals  = res.phi_grid_deg[:, 0]
    lat_vals  = res.theta_grid_deg[0, :]
    SBElon, SBElat = compute_subearth_track(cr_num)

    speed_interp = sample_interpolated(speed_wsaplus, lon_vals, lat_vals, SBElon, SBElat)
    lon_nearest, lat_nearest, speed_nearest = sample_nearest(speed_wsaplus, lon_vals, lat_vals, SBElon, SBElat)
    # Get sorting indices
    lon_nearest = np.array(lon_nearest); speed_nearest = np.array(speed_nearest)
    sort_idx = np.argsort(lon_nearest)
    lon_sorted = lon_nearest[sort_idx]
    speed_sorted = speed_nearest[sort_idx]

    # Get sorting indices
    SBElon = np.array(SBElon); speed_interp = np.array(speed_interp)
    sort_idx = np.argsort(SBElon)
    SBElon_sorted = SBElon[sort_idx]
    speed_interp_sorted = speed_interp[sort_idx]

    lon_grid_360 = np.arange(1, 361)
    speed_nearest_360 = np.interp(lon_grid_360, lon_sorted, speed_sorted)
    speed_interp_360 = np.interp(lon_grid_360, SBElon_sorted, speed_interp_sorted)
    
    plt.figure(figsize=(10,5))
    plt.plot(lon_grid_360, speed_interp_360, label="Interpolated")
    plt.plot(lon_grid_360, speed_nearest_360, label="Nearest grid cell")
    plt.xlabel("Carrington Longitude (deg)")
    plt.ylabel("Solar wind speed (km/s)")
    plt.legend()
    plt.title("Sub-Earth speed profile")
    plt.savefig(outdir / f"huxt_input_speed_profile_{event}.png")
    plt.close()
    
    plt.figure(figsize=(10,4))
    plt.plot(lon_grid_360, speed_interp_360 - speed_nearest_360)
    plt.xlabel("Carrington Longitude (deg)")
    plt.ylabel("Speed difference (km/s)")
    plt.title("Interpolation − Nearest")
    plt.savefig(outdir / f"huxt_input_speed_difference_{event}.png")
    plt.close()
    
    plt.figure(figsize=(8,5))
    plt.pcolormesh(lon_vals, lat_vals, speed_wsaplus.T)
    plt.scatter(SBElon, SBElat, s=3, c="red", label="Sub-Earth track")
    plt.scatter(lon_nearest, lat_nearest, s=3, c="black", label="Nearest grid cells")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Sub-Earth track on WSA map")
    plt.colorbar(label="Speed km/s")
    plt.savefig(outdir / f"huxt_input_subearth_track_{event}.png")
    plt.close()

    return lon_grid_360, speed_nearest_360

def prepare_background(cme_onset, outdir=None, checkpoint_path=None, download=False, event=None):
    """
    Prepare the background input for HUXt by downloading GONG magnetograms, finding the closest map, running WSA+, and mapping the speed to the sub-Earth track.
    Input:
    - cme_onset: CME onset time (string in ISO format)
    - outdir: directory to save the output files and plots
    - checkpoint_path: path to the WSA+ checkpoint file
    - download: whether to download GONG magnetograms (default: False)
    - event: event name for labeling outputs
    Output:
    - cr_num: Carrington rotation number of the closest GONG map
    - obs_time: observation time of the closest GONG map
    - v_boundary: speed map interpolated to the sub-Earth track, ready to be used as input for HUXt
    """
    # Ensure the output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get the CME time (example: 2017-09-06 14:00:38)
    t_cme = Time(cme_onset, scale="utc") #Time("2024-01-01T00:00:00")
    print ("CME at:", t_cme)

    # Step 2: Download GONG magnetogram (if not already downloaded)     
    t_start_mag = t_cme - 6 * u.hour #t_cme is an astropy Time object
    t_end_mag   = t_cme - 0 * u.hour

    if download: 
        download_gong_mag(t_start_mag, t_end_mag, outdir=outdir)

    # Step 3: Find the closest GONG magnetogram 
    gong_map, obs_time, dt_mag_cme = find_closest_map(
        t_cme, map_dir=outdir
    )
    print(f"Closest GONG map found at {obs_time} (Δt = {dt_mag_cme.to(u.hour)})")
    cr_num = carrington_rotation_number(gong_map.date)
    print ('GONG map.date:', gong_map.center, 'CR number:', cr_num)
    #print ('GONG map.date:', gong_map.meta.key(), 'CR number:', cr_num)

    mid_cr = gong_map.meta.get('CRCENTER', 'Default Value')
    print ('CR center from metadata:', mid_cr)


    # Step 4: Generate WSA+ speed map (cached)
    speed_map_path = outdir / f"wsaplus_speed_map_{event}.npz"

    if not speed_map_path.exists():
        speed_wsaplus = run_wsaplus(gong_map, outdir=outdir, checkpoint_path=checkpoint_path, event=event)
        _, speed_360 = map_input_huxt(speed_wsaplus, outdir=outdir, event=event, cr_num=cr_num)
    else:
        data = np.load(speed_map_path, allow_pickle=True)
        speed_wsaplus = data["speed_map"].item()
        _, speed_360 = map_input_huxt(speed_wsaplus, outdir=outdir, event=event, cr_num=cr_num)

    
    # 3. Return boundary
    np.savez(outdir / f"v_boundary_{event}.npz", speed_map=speed_360)
    v_boundary = speed_360 * (u.km / u.s)

    return cr_num, obs_time
    
def write_event_config(filename, initial_theta=None, cr_num=None):
    """
    Write the per-event seed config (event_config.yaml).
    Input:
    - filename: path to save the config file
    - initial_theta: seed Cone-CME parameter vector [inject_hour, longitude, latitude, width, speed]
    - cr_num: Carrington rotation number of the selected magnetogram
    Output:
    - A YAML config file saved at the specified filename
    The downstream GP/HUXt workflow reads only initial_theta and cr_num.
    """
    if initial_theta is None:
        initial_theta = [1.0, 0.0, 0.0, 60.0, 1000.0]

    config = {
        "initial_theta": list(map(float, initial_theta)),
        "cr_num": float(cr_num),
    }

    # Write YAML
    filename = Path(filename)
    with filename.open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Config file written to: {filename}")

def process_event(event_dict, output_root, checkpoint_path, download=True, force_config=False, sanity_plot=True):
    """Process a single CME event by preparing the background and writing the seed config.
    Input:
    - event_dict: dictionary containing event information (event name, CME onset time, CME time at 0.1 AU, longitude, latitude, width, speed)
    - output_root: base directory for saving outputs
    - download: whether to download GONG magnetograms (default: True)
    Output: - Prepared background files and event_config.yaml for the event
    """
    event = event_dict["event"]
    cme_onset = event_dict["cme_onset"]
    cme_0p1_au = event_dict["cme_0p1_au"]

    longitude = event_dict["longitude"]
    latitude = event_dict["latitude"]
    width = event_dict["width"]
    speed = event_dict["speed"]

    print(f"\n===== Processing event: {event} =====")

    output_root = Path(output_root)
    outdir = output_root / event
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Background preparation ---
    cr_num, closest_mag_time = prepare_background(
        cme_onset,
        outdir=outdir,
        checkpoint_path=checkpoint_path,
        download=download,
        event=event,
    )

    print("Closest mag time:", closest_mag_time)

    # --- Compute injection time ---
    inject_time = Time(cme_0p1_au) - Time(closest_mag_time)
    inject_hour = inject_time.to(u.hour).value

    print("Injecting CME after:", inject_time.to(u.hour))
    print("Inject hour value:", inject_hour)

    # --- Load v_boundary ---
    data = np.load(outdir / f"v_boundary_{event}.npz")
    vboundary = data["speed_map"] * (u.km / u.s)

    # --- HUXt config ---
    HUXT_KWARGS = dict(
        v_boundary=vboundary,
        latitude=0 * u.deg,
        cr_num=cr_num,
        frame="sidereal",
        simtime=10 * u.day,
        dt_scale=4,
    )

    # --- Build theta ---
    theta = (inject_hour, longitude, latitude, width, speed,)
    print("Theta:", theta)

    # --- Run simulation (optional sanity check) ---
    if sanity_plot:
        huxt_time, huxt_speed, starttime, endtime = rhf.run_huxt_sim(
            theta,
            HUXT_KWARGS,
            plot=True,
            outdir=str(outdir) + "/",
            event=event,
        )

    # --- Write the per-event seed config ---
    config_file = outdir / "event_config.yaml"
    if config_file.exists() and not force_config:
        print(f"Skipping {event} (already processed)")
        return

    write_event_config(
        filename=config_file,
        initial_theta=[inject_hour, longitude, latitude, width, speed],
        cr_num=cr_num,
    )

    print(f"Finished event: {event}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", type=Path, default=DEFAULT_EVENTS_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to wsaplus.pt. Defaults to <output-root>/wsaplus.pt.",
    )
    parser.add_argument(
        "--event",
        nargs="+",
        default=["all"],
        help="Event name(s) from the CSV, or all.",
    )
    parser.add_argument("--no-download", action="store_true", help="Reuse existing GONG FITS files only.")
    parser.add_argument("--force-config", action="store_true", help="Overwrite existing event_config.yaml files.")
    parser.add_argument("--skip-sanity-plot", action="store_true", help="Skip the seed HUXt run and plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint_path or args.output_root / "wsaplus.pt"
    events = load_events(args.events_file, args.event)
    for event_dict in events:
        process_event(
            event_dict,
            args.output_root,
            checkpoint_path,
            download=not args.no_download,
            force_config=args.force_config,
            sanity_plot=not args.skip_sanity_plot,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
