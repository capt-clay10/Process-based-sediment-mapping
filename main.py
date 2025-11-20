'''
This code generates sediment distribution maps for Delft3D-4 outputs

Required files 
    - Accumulated flux files for each sediment  (.mat file = v7.3/hdf5)
    - Hydrodynamic grid file (.mat file = v7.3/hdf5)
Optional files 
    - Bed shear stress map files from Delft3D-4 output (.mat file = v7.3/hdf5)


Notes: 

This code focuses on the ASF implementation to generate a sediment distribution
map, however the user is free to try other operations and analysis methods as 
well, it should be noted that following research ASF approach works most
promisingly.

Currently the code only works for grain sizes from 3.5Phi to 0.5Phi
to modify, modifications must be done in function:
sp_interp : the interpolation function 

Author: Clayton Soares
    
'''
# %% Import packages
from __future__ import annotations
from threading import Lock
import multiprocessing
import h5py
import re
import os
import gc
import glob
import numpy as np
import pandas as pd
import psutil
import scipy.stats as ss
from tqdm import tqdm
import concurrent.futures
import time
from scipy.interpolate import UnivariateSpline
from scipy.stats import scoreatpercentile
from scipy.interpolate import interp1d
from datetime import datetime
from typing import Tuple, Optional

# Using IPython magic in script is optional; comment out if not running in notebook
# get_ipython().run_line_magic('matplotlib', 'qt')


# %% Functions


# ------------------------------Auxiliary functions---------------------------#

def extract_number(s):
    """
    Extracts a number from a string matching the pattern: '{some_digits}_perc'.
    Returns an integer or None if no match is found.
    """
    match = re.search(r'(\d+)_perc', s)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_sort_key(filename):
    """
    Extract a numeric key from the filename for sorting by grain size.
    Example: '300um_something.mat' -> 300
    """
    match = re.search(r"(\d+)um", filename)
    if match:
        return int(match.group(1))
    else:
        return filename  # Fallback if pattern does not match numeric grain size


def find_files(root_dir, file_extension):
    """
    Finds all files matching 'file_extension' in 'root_dir' (recursively),
    returns a sorted list by numeric grain size if found in the filename.
    """
    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in glob.glob(os.path.join(dirpath, file_extension)):
            matching_files.append(file)
    matching_files = sorted(matching_files, key=extract_sort_key)
    return matching_files


def read_units_names(file_path):
    """
    Reads 'Units' and 'Name' from the .mat file under '/data/Units' 
    and '/data/Name'.
    """
    with h5py.File(file_path, 'r') as file:
        units_data = file['data/Units'][:]
        name_data = file['data/Name'][:]
        units = ''.join(chr(i) for i in units_data.flatten())
        name = ''.join(chr(i) for i in name_data.flatten())
        return units, name


def _compute_n_trim(val: np.ndarray, trim_fraction: float) -> np.ndarray:
    """Return the number of samples kept after *trim_fraction* was removed on
    each side of the distribution (axis=2)."""
    sorted_val = np.sort(val, axis=2)
    lower = np.nanpercentile(
        sorted_val, trim_fraction * 100, axis=2, keepdims=True)
    upper = np.nanpercentile(
        sorted_val, (1 - trim_fraction) * 100, axis=2, keepdims=True)
    mask = (sorted_val >= lower) & (sorted_val <= upper)
    return np.sum(mask, axis=2)


def combine_xyz_files(files):
    """
    Example aggregator for .xyz files (space-separated).
    Reads last column from each file, concatenates into DataFrame.
    """
    combined_data = []
    for file in files:
        data = pd.read_csv(file, sep=' ', skiprows=1, header=None)
        combined_data.append(data.iloc[:, -1])
    return pd.concat(combined_data, axis=1)


def combine_plot_files(files):
    """
    Example aggregator for .xyz-like files but comma-separated.
    Reads last column from each file, concatenates into DataFrame.
    """
    combined_data = []
    for file in files:
        data = pd.read_csv(file, sep=',', skiprows=0, header=None)
        combined_data.append(data.iloc[:, -1])
    return pd.concat(combined_data, axis=1)

def remove_pattern(s, pattern):
    """Utility to remove 'pattern' from string s."""
    return re.sub(pattern, '', s).strip()


def compute_tau_c_soulsby(grain_sizes_m,
                          rho_s=2650,
                          rho=1000,
                          g=9.81,
                          nu=1e-6):
    """
    Returns the dimensional critical shear stress [Pa]
    using Soulsby's (1997) curve-fit to Shields.

    :param grain_sizes_m: array-like, [m]
    :param rho_s: sediment density [kg/m^3]
    :param rho: fluid density [kg/m^3]
    :param g: gravitational acc [m/s^2]
    :param nu: kinematic viscosity [m^2/s]
    :return: tau_c array [Pa]
    """

    s = rho_s / rho
    d_star = grain_sizes_m * ((g * (s - 1)) / (nu**2)) ** (1/3)

    # Soulsby approximate formula for dimensionless theta_c:
    theta_c = 0.30 * (1 + 1.2 * d_star) ** (-1) \
        + 0.055 * (1 - np.exp(-0.020 * d_star))

    # Convert dimensionless -> dimensional
    tau_c = theta_c * (rho_s - rho) * g * grain_sizes_m
    return tau_c


INVALID_CHARS = r'<>:"/\|?*'

def safe_filename(s: str, repl: str = "_") -> str:
    s = s.strip()
    return re.sub(f"[{re.escape(INVALID_CHARS)}]", repl, s)

# ------------------------------Main functions--------------------------------#


def operate_mat(
    file_name: str,
    operation: int,
    choice: int,
    analysis: int,
    *,
    exceed_pct: float = 25.0,
    bss_array: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, str, str, Optional[np.ndarray]]:
    """
    Compute grain-size-specific flux (or CFR) statistics from 
    a Delft3D MAT file.

    Parameters
    ----------
    file_name   : path to *.mat* file that contains /data/Val and, 
    for op 9, /data/BSS
    
    operation   : 1-9  (see calling script)
    choice      : 1 = positive, 2 = negative, else = total
    analysis    : 1-6  direct flux statistic   |  7 = CFR (pos / neg − 1)
    exceed_pct  : percentile threshold for operations 8 & 9 (default 25 %)

    Returns
    -------
    val_in  : 2-D array (nx, ny)
    title   : descriptive title string
    units   : units string from the file
    N_trim  : counts used in trimmed mean (op 6) else None
    """
    # Open file and pull meta + arrays
    gs_match = re.search(r"(\d+)um", file_name)
    grain_size = int(gs_match.group(1)) if gs_match else "unknown"

    with h5py.File(file_name, "r") as f:
        dset = f["data"]
        units = "".join(chr(i) for i in dset["Units"][:].ravel())
        name  = "".join(chr(i) for i in dset["Name"] [:].ravel())
        title_mid_end = "".join(c for c in name if c.isalpha() or c.isspace()).lower()

        # Expect Val shape = (nx, ny, nt)
        val_real = dset["Val"][:]


    # Memory-aware copy to float32 if RAM is tight
    BYTES_PER_GB = 1024 ** 3
    avail_gb = psutil.virtual_memory().available / BYTES_PER_GB
    req_gb   = val_real.nbytes / BYTES_PER_GB
    dtype    = np.float32 if req_gb > 0.9 * avail_gb else val_real.dtype
    val_c    = val_real.astype(dtype, copy=True) 



    # Sign selection  (use NaN to REMOVE the opposite sign from stats)
    # We keep only the target sign and set everything else to NaN so that
    # nan-aware percentiles/means ignore those samples.
    
    if choice == 1:      # positive flux only
        val = np.where(val_c > 0, val_c, np.nan).astype(float, copy=False)
        title_mid = "positive"
    elif choice == 2:    # negative flux only (use magnitude)
        val = np.where(val_c < 0, np.abs(val_c), np.nan).astype(float, copy=False)
        title_mid = "negative"
    else:                # total flux (no sign filtering)
        val = val_c.astype(float, copy=True)
        title_mid = "total"

    # convert to mg m^-2 s^-1 for internal computations (NaNs preserved)
    val *= 1000.0



    # Helper for op 8 (exceedance mean)
    def _exceedance_mean(arr: np.ndarray) -> np.ndarray:
        thr   = np.nanpercentile(arr, exceed_pct, axis=2, keepdims=True)
        mask  = arr >= thr
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.where(mask, arr, np.nan), axis=2)



    # Analysis 1-6: direct flux statistics
    N_trim: Optional[np.ndarray] = None
    if analysis in {1, 2, 3, 4, 5, 6}:
        if   operation == 1:
            val_in, title_start = np.nanmean(val, axis=2), "Mean"
        elif operation == 2:
            val_in, title_start = np.nanpercentile(val, 95, axis=2), "95th Percentile"
        elif operation == 3:
            val_in, title_start = np.nanmin(val, axis=2), "Minimum"
        elif operation == 4:
            val_in, title_start = np.nanmax(val, axis=2), "Maximum"
        elif operation == 5:
            val_in, title_start = np.nanmedian(val, axis=2), "Median"
        elif operation == 6:
            # compute IQR bounds with NaN-aware percentiles
            lower = np.nanpercentile(val, 25, axis=2, keepdims=True)
            upper = np.nanpercentile(val, 75, axis=2, keepdims=True)
            mask  = (val >= lower) & (val <= upper)
            with np.errstate(invalid="ignore"):
                val_in = np.nanmean(np.where(mask, val, np.nan), axis=2)
            # count how many timesteps (non-NaN) contributed within IQR
            N_trim = np.sum(mask & ~np.isnan(val), axis=2)
            title_start = "Trimmed Mean"
        elif operation == 7:
            val_in, title_start = np.nanpercentile(val, 75, axis=2), "75th Percentile"
        elif operation == 8:
            val_in   = np.nan_to_num(_exceedance_mean(val), nan=0.0)
            title_start = f"Mean Above {exceed_pct:g}th Perc"
        elif operation == 9:
            if bss_array is None:
                raise ValueError("operate_mat: bss_array must be supplied for op 9")
            # Check against the time dimension of the original series
            if bss_array.shape[2] != val_c.shape[2]:
                raise ValueError("Time dimension mismatch between Val and BSS")
            bss_thr = np.nanpercentile(bss_array, exceed_pct, axis=2, keepdims=True)
            mask    = bss_array >= bss_thr
            with np.errstate(invalid="ignore"):
                val_in = np.nanmean(np.where(mask, val, np.nan), axis=2)
            val_in   = np.nan_to_num(val_in, nan=0.0)
            title_start = f"Mean Flux | BSS>{exceed_pct:g}th Perc"
        else:
            raise ValueError("operation must be 1–9 for analysis ≠ 7")

        # back to kg/m2
        val_in /= 1000.0



    # Analysis 7: CFR  (supports operations 1–9)
    elif analysis == 7:
        # sign-conditioned magnitudes with NaNs elsewhere
        pos_flux = np.where(val_c > 0, val_c, np.nan) * 1000.0
        neg_flux = np.where(val_c < 0, np.abs(val_c), np.nan) * 1000.0

        if   operation == 1:
            pos_stat, neg_stat, title_start = np.nanmean(pos_flux, 2), np.nanmean(neg_flux, 2), "Mean CFR"
        elif operation == 2:
            pos_stat, neg_stat, title_start = np.nanpercentile(pos_flux, 95, 2), np.nanpercentile(neg_flux, 95, 2), "95th Perc CFR"
        elif operation == 3:
            pos_stat, neg_stat, title_start = np.nanmin(pos_flux, 2), np.nanmin(neg_flux, 2), "Minimum CFR"
        elif operation == 4:
            pos_stat, neg_stat, title_start = np.nanmax(pos_flux, 2), np.nanmax(neg_flux, 2), "Maximum CFR"
        elif operation == 5:
            pos_stat, neg_stat, title_start = np.nanmedian(pos_flux, 2), np.nanmedian(neg_flux, 2), "Median CFR"
        elif operation == 6:
            pos_stat, neg_stat, title_start = ss.trim_mean(pos_flux, 0.25, 2), ss.trim_mean(neg_flux, 0.25, 2), "Trimmed Mean CFR"
        elif operation == 7:
            pos_stat, neg_stat, title_start = np.nanpercentile(pos_flux, 75, 2), np.nanpercentile(neg_flux, 75, 2), "75th Perc CFR"
        elif operation == 8:
            def _exc(arr):
                thr = np.nanpercentile(arr, exceed_pct, axis=2, keepdims=True)
                msk = arr >= thr
                with np.errstate(invalid="ignore"):
                    return np.nanmean(np.where(msk, arr, np.nan), axis=2)
            pos_stat = np.nan_to_num(_exc(pos_flux), nan=0.0)
            neg_stat = np.nan_to_num(_exc(neg_flux), nan=0.0)
            title_start = f"CFR Mean > {exceed_pct:g}th Perc"
        elif operation == 9:
            if bss_array is None:
                raise ValueError("operate_mat: bss_array must be supplied for op 9")
            if bss_array.shape[2] != val_c.shape[2]:
                raise ValueError("Time dimension mismatch between Val and BSS")
            bss_thr = np.nanpercentile(bss_array, exceed_pct, axis=2, keepdims=True)
            mask    = bss_array >= bss_thr
            with np.errstate(invalid="ignore"):
                pos_stat = np.nanmean(np.where(mask, pos_flux, np.nan), axis=2)
                neg_stat = np.nanmean(np.where(mask, neg_flux, np.nan), axis=2)
            pos_stat  = np.nan_to_num(pos_stat, nan=0.0)
            neg_stat  = np.nan_to_num(neg_stat, nan=0.0)
            title_start = f"CFR Mean | BSS>{exceed_pct:g}th Perc"
        else:
            raise ValueError("operation must be 1–9 for analysis 7")

        # avoid divide-by-zero in CFR
        neg_stat = np.where(neg_stat < 1e-12, 1e-12, neg_stat)
        val_in   = np.maximum(pos_stat / neg_stat, 1) - 1
        title_mid = "total"

    else:
        raise ValueError("analysis must be 1–7")

    title = f"{title_start} {title_mid} {title_mid_end} {grain_size}um"
    return val_in, title, units, N_trim




def sp_interp(df_in, grain_sizes, percentile):
    """
    Function for computing a user-specified percentile (e.g., D50).
    Uses interpolation on a phi scale, each grain size percentage is distributed
    equally across the interval before calculating percentiles.
    but included for completeness.
    """
    df = df_in.copy()
    column_name = f'D{percentile}'
    d_interp_values = []

    grain_sizes = np.array(grain_sizes, dtype=float)
    grain_sizes_mm = grain_sizes / 1000.0
    phi_values = np.round(-np.log2(grain_sizes_mm), 1)

    phi_intervals = [(3.5, 3.0), (3.0, 2.5), (2.5, 2.0),
                     (2.0, 1.5), (1.5, 1.0), (1.0, 0.5)]
    phi_to_grain_mapping = {}

    for phi, grain_size in zip(phi_values, grain_sizes):
        for interval in phi_intervals:
            phi_max, phi_min = interval
            if phi_max >= phi > phi_min:
                phi_to_grain_mapping[interval] = grain_size
                break

    smallest_grain_size = grain_sizes[0]
    smallest_interval = phi_intervals[0]
    if smallest_interval not in phi_to_grain_mapping:
        phi_to_grain_mapping[smallest_interval] = smallest_grain_size
        print(f"{smallest_grain_size} µm assigned to Phi interval {smallest_interval}")

    for _, row in df.iterrows():
        fine_phi_values = np.linspace(3.5, 0.5, 60)
        fine_distribution = np.zeros_like(fine_phi_values)

        for (phi_min, phi_max), grain_size in phi_to_grain_mapping.items():
            grain_size_str = str(int(grain_size))
            if grain_size_str in row:
                availability = row[grain_size_str]
                mask = (fine_phi_values >= phi_max) & (
                    fine_phi_values <= phi_min)
                if np.any(mask) and np.sum(mask) > 0:
                    fine_distribution[mask] += availability / np.sum(mask)

        cumulative_distribution = np.cumsum(fine_distribution)
        if cumulative_distribution[-1] > 0:
            cumulative_distribution_percent = (
                cumulative_distribution / cumulative_distribution[-1]) * 100
        else:
            cumulative_distribution_percent = np.zeros_like(
                cumulative_distribution)

        try:
            if len(np.unique(cumulative_distribution_percent)) > 1:
                d_phi = np.interp(
                    percentile, cumulative_distribution_percent, fine_phi_values)
                d_micron = 2 ** (-d_phi) * 1000
            else:
                d_micron = np.nan
        except ValueError:
            d_micron = np.nan

        d_interp_values.append(round(d_micron, 2))

    df[column_name] = d_interp_values
    return df


def process_flux_data(
    analysis,
    grain_sizes_w,
    factor_sed_mass,
    val_pos_list=None,
    val_neg_list=None,
    cfr_list=None,
    pos_title=None,
    neg_title=None,
    cfr_title=None,
    N_trim_pos_list=None,
    N_trim_neg_list=None,
    thresh=None,
):
    """
    Combine positive and negative flux statistics (and optional CFR fields)
    across grain sizes, for different analysis modes.

    Parameters
    ----------
    analysis : int
        1 = Split depositional / erosional (threshold in 'thresh')
        2 = Positive flux only
        3 = Negative flux only
        4 = Residual depositional (pos - neg, clipped >= 0)
        5 = Geometric mean of |pos| and |neg|
        6 = Total transport
            - if N_trim_* are given (e.g. op=6): ASF = μ⁺ N⁺ + μ⁻ N⁻
            - otherwise: |pos| + |neg|
        7 = CFR (cumulative flux ratio)

    grain_sizes_w : list or array
        Grain sizes used for shear-stress weighting (µm).
    factor_sed_mass : float
        0 → no shear-stress weighting; >0 → multiply each grain by
        Soulsby tau_c / max(tau_c) * factor_sed_mass.
    val_pos_list, val_neg_list : list[np.ndarray]
        Per-grain 2-D fields of positive and negative flux statistics.
    cfr_list : list[np.ndarray]
        Per-grain CFR fields (for analysis = 7).
    pos_title, neg_title, cfr_title : list[str]
        Corresponding titles.
    N_trim_pos_list, N_trim_neg_list : list[np.ndarray] or None
        Per-grain counts of time steps that entered the statistic.
        For op = 6 these are the counts of time steps inside the IQR
        used for the trimmed mean. Needed for ASF (analysis = 6).
    thresh : float or None
        Grain-size threshold for analysis = 1.

    Returns
    -------
    dataset : np.ndarray or None
        Combined 3-D array (nx, ny, n_grains).
    dataset_title : list[str]
        Titles for each grain-size slice.
    """

    if val_pos_list is None:
        val_pos_list = []
    if val_neg_list is None:
        val_neg_list = []
    if cfr_list is None:
        cfr_list = []
    if pos_title is None:
        pos_title = []
    if neg_title is None:
        neg_title = []
    if cfr_title is None:
        cfr_title = []

    # Time-step weighting availability (needed for ASF)
    use_time_step_weights = (
        isinstance(N_trim_pos_list, list)
        and len(N_trim_pos_list) > 0
        and all(isinstance(arr, np.ndarray) for arr in N_trim_pos_list)
        and isinstance(N_trim_neg_list, list)
        and len(N_trim_neg_list) > 0
        and all(isinstance(arr, np.ndarray) for arr in N_trim_neg_list)
    )

    if use_time_step_weights:
        # shape (nx, ny, n_grains)
        N_trim_pos_all = np.stack(N_trim_pos_list, axis=2)
        N_trim_neg_all = np.stack(N_trim_neg_list, axis=2)
        N_trim_total = N_trim_pos_all + N_trim_neg_all
        # avoid zeros where both signs have no central events
        N_trim_total[N_trim_total == 0] = 1
    else:
        N_trim_pos_all = None
        N_trim_neg_all = None
        N_trim_total = None

    # Shear-stress weighting (Soulsby) across grain sizes
    grain_sizes_m = np.array(grain_sizes_w, dtype=float) * 1e-6
    tau_ci = compute_tau_c_soulsby(grain_sizes_m)

    if factor_sed_mass == 0:
        weights = np.ones_like(tau_ci)
    else:
        tau_ci_norm = tau_ci / tau_ci.max()
        weights = tau_ci_norm * factor_sed_mass

    # reshape to broadcast over (nx, ny, n_grains)
    weights_reshaped = weights.reshape(1, 1, -1)


    # Stack flux arrays for all grain sizes
    pos_all = np.stack(val_pos_list, axis=2) if len(val_pos_list) > 0 else None
    neg_all = np.stack(val_neg_list, axis=2) if len(val_neg_list) > 0 else None
    cfr_all = np.stack(cfr_list,     axis=2) if len(cfr_list) > 0 else None

    # For analyses 1–5 we retain the original time-weighted logic
    if use_time_step_weights and (pos_all is not None) and (neg_all is not None):
        frac_pos = N_trim_pos_all / N_trim_total
        frac_neg = N_trim_neg_all / N_trim_total
        time_weighted_pos = frac_pos * pos_all
        time_weighted_neg = frac_neg * neg_all
    else:
        time_weighted_pos = pos_all
        time_weighted_neg = neg_all


    # Analysis-specific combination
    dataset = None
    dataset_title: list[str] = []

    if analysis == 1:
        # Split into depositional / erosional based on 'thresh'
        split_values = []
        split_titles = []
        source_pos = time_weighted_pos if time_weighted_pos is not None else pos_all
        source_neg = time_weighted_neg if time_weighted_neg is not None else neg_all

        if (source_pos is None) or (source_neg is None):
            return None, []

        for i, title in enumerate(pos_title):
            m = re.search(r"(\d+)um\b", title)
            if not m:
                continue
            gs = int(m.group(1))
            if thresh is None:
                raise ValueError("analysis=1 requires 'thresh' to be set.")
            if gs <= thresh:
                split_values.append(source_pos[:, :, i])
                split_titles.append(title.replace("positive", "Deposition"))
            else:
                split_values.append(source_neg[:, :, i])
                split_titles.append(neg_title[i].replace("negative", "Erosion"))

        if not split_values:
            return None, []
        dataset = np.stack(split_values, axis=2)
        dataset_title = split_titles

    elif analysis == 2:
        # Positive flux only
        dataset = time_weighted_pos
        dataset_title = pos_title

    elif analysis == 3:
        # Negative flux only
        dataset = time_weighted_neg
        dataset_title = neg_title

    elif analysis == 4:
        # Residual depositional = (pos - neg), clipped >= 0
        if (time_weighted_pos is not None) and (time_weighted_neg is not None):
            diff_pos_all = time_weighted_pos - time_weighted_neg
            diff_pos_all[diff_pos_all < 0] = 0.0
            diff_pos_all[np.abs(diff_pos_all) < 1e-6] = 0.0
            dataset = diff_pos_all
            dataset_title = [
                s.replace("positive", "Weighted Residual Depositional Flux")
                for s in pos_title
            ]

    elif analysis == 5:
        # Geometric-mean transport (emphasises co-occurrence)
        if (time_weighted_pos is not None) and (time_weighted_neg is not None):
            P = np.maximum(time_weighted_pos, 0.0)
            N = np.maximum(time_weighted_neg, 0.0)
            eps = 1e-12
            dataset = np.sqrt(P * N + eps)
            dataset_title = [
                s.replace("negative", "Geometric Mean Transport")
                for s in neg_title
            ]

    elif analysis == 6:

#---------ASF / Total transport ----------------------------------------------#

        if (pos_all is not None) and (neg_all is not None):
            if use_time_step_weights:
                # ASF: central trimmed means μ⁺, μ⁻ scaled by counts N⁺, N⁻
                # per cell and grain size → gross exchange of "typical" events
                dataset = pos_all * N_trim_pos_all + neg_all * N_trim_neg_all
                dataset_title = [
                    s.replace("negative", "ASF central gross exchange")
                    for s in neg_title
                ]
            else:
                # No N_trim available → fall back to magnitude-based
                # total transport (|pos| + |neg|)
                P = np.maximum(pos_all, 0.0)
                N = np.maximum(neg_all, 0.0)
                dataset = P + N
                dataset_title = [
                    s.replace("negative", "Weighted Total Transport")
                    for s in neg_title
                ]

    elif analysis == 7:
        # CFR (cumulative flux ratio)
        if cfr_all is not None:
            dataset = cfr_all
            dataset_title = [
                s.replace("Cumulative Flux Ratio", "Weighted Cumulative Flux Ratio")
                for s in cfr_title
            ]

    else:
        raise ValueError("Invalid analysis method (1..7).")


    # Final checks and shear-stress weighting
    if dataset is None:
        return None, []

    # Apply Soulsby-based shear-stress weights (if enabled)
    dataset = dataset * weights_reshaped

    return dataset, dataset_title

# %% User input 

if __name__ == "__main__":
    
    st = time.time()
    now = datetime.now()
    print(f"Script started at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # -----------------------------------------------------------------------#
    '''
    The delft3D-4 files should be outputed with the following naming convention
    grainsize_model_type.mat, example: 105um_asf.mat
    currently the code only supports six grain size inputs
    representing 3.5 to 0.5 phi.
    
    the delft3d-4 hydrodynamic grid should be exported for one time step but
    the k layer should be changed to 1 instead of all while exporting in the 
    mat format.
    
    '''
    
    # -----------------------------------------------------------------------#
    main_path = 'E:/delft3d_python/Model_based_methods/m2_ASF_phi_soulsby/'
    os.chdir(main_path)

    # Model type for file identification
    model_type = 'sf'
    
    um_files = find_files(main_path, f"*um_{model_type}.mat")
    
    hyd_grid = find_files(main_path, 'hyd_grid_nf.mat')

    with h5py.File(hyd_grid[0], 'r') as file:
        data = file['data']
        x = data['X'][:, :]
        y = data['Y'][:, :]


    bss_file = 'E:/delft3d_python/Model_based_methods/bss_rep_period.mat'
    
    with h5py.File(bss_file, "r") as fb:
        # adapt the path if the array is stored somewhere else inside the MAT
        bss_all = fb["data"]["Val"][:]          # shape (nx, ny, nt)
        # convert straight to float32 to halve RAM, optional
        bss_all = bss_all.astype(np.float32)
        print("BSS loaded:", bss_all.shape, bss_all.dtype)

    # ------------------------------------------------------------------------
    # USER PARAMETERS
    # ------------------------------------------------------------------------
    # Descriptor for each individual flux file 
    
    '''
    here you choose how to compress the full time series of flux files for 
    each grain size file, so op = 1 would mean all values per grid cell are 
    averaged across time of simulation.
    
    however in this stage the positive and negative fluxes are separately
    operated upon
    
    '''
    op = 6  # choose the descriptor

    # 1= airthmetic average
    # 2= 95th percentil
    # 3= min
    # 4= max
    # 5= median
    # 6= ASF <----------------------------------------------------------------#
    # 7= 75th percentile
    # 8= Exceedance average , only flux values about a certain percentile are averaged
    # 9= BSS exceedance, only time points above a bss percentile are averaged
    
    exceed_pct = 25.0  # exceedence percentile for op 8 and 9

    # ------------------------------------------------------------------------#
    # Descriptor for positive and negative fluxes 
    '''
    here you choose how to combine the positive and negative operated
    flux values
    
    '''     
    analysis = 6 # Choose analysis method

    # 1= Combined flux (105-210[depositive] and 300+[erosive])
    # 2 = Only positive flux
    # 3 = Only negative flux
    # 4 = Difference flux abs pos - abs neg - annual residual depositive flux
    # 5 = Geometric mean - if both high then value high
    # 6= ASF <----------------------------------------------------------------#
    # 7 = Cumulative flux ratio

    thresh = 300  # choose grain size
    '''
    # experimental approach
    This is the threshold for analysis method 1
    choose from which grain size on the grains are more erosion based
    for eg, 105 is more deposition based and 600 is more erosion based
    this method splits the approach so under the threshold you only pick
    pos flux and above neg flux
    so you basically add the absolute values of the pos and neg fluxes
    
    '''

    # ------------------------------------------------------------------------#
    # Adding weights
    '''
    Here you can choose if you want to apply weights to the individual sediment
    operated and analysed fluxes before creating the sediment distribution,
    just give the list of your grain sizes and choose the weight factor
    Soulsby formulation is used for generating weights.
    
    '''
    grain_sizes_w = [105, 150, 210, 300, 420, 600]

    # grain sizes for weight calculation
    # Here the weights are calculated based on Van Rijns parameter of critical
    # shear stress.

    factor_sed_mass = 0  # Scaling factor for weight

    # 0 implies no scaling_factor
    # >=1 weights applied

    # ------------------------------------------------------------------------#
    # Choose if you want to export D50 map at the end 
    
    d50_on = 1  # choose if d50 should be calcualted

    # 0 = off
    # 1 = valid for [105, 150, 210, 300, 420, 600]
    # 2 = valid for [130, 160, 200, 255, 360, 500]
    # for custom combinations edit the code below

    choice_d50 = 1  # choose what range D50 should be calculated for

    # 1 = 105-600
    # 2 = 150-420
    # 3 = 105-420
    # 4 = 130-360
    # 5 = 130-500

    # %% Main code------------------------------------------------------------#

    val_pos_dict = {}
    val_neg_dict = {}
    N_trim_pos_dict = {}
    N_trim_neg_dict = {}
    pos_title_dict = {}
    neg_title_dict = {}
    cfr_dict = {}
    cfr_title_dict = {}

    # For reduced memory usage, you might want to reduce the number of concurrent workers.
    max_worker = 4  # reduced from 22
    batch_size = 2  # process files in batches of 10
    lock = Lock()

    def process_file(mat_path: str) -> Tuple[
            Optional[np.ndarray],  # val_pos
            Optional[str],         # title_pos
            Optional[np.ndarray],  # val_neg
            Optional[str],         # title_neg
            Optional[np.ndarray],  # cfr
            Optional[str],         # title_cfr
            Optional[np.ndarray],  # N_trim_pos
            Optional[np.ndarray]   # N_trim_neg
    ]:
        """
        Processes one .mat file: extracts pos & neg flux arrays plus N_trim if op==6, etc.
        """
        val_pos = title_pos = val_neg = title_neg = None
        cfr = title_cfr = None
        N_trim_pos = N_trim_neg = None

        try:
            # Positive / negative flux statistics (analysis 1–6 require them;
            # 7 ignores them but we keep the branch symmetrical).
            if analysis in {1, 2, 3, 4, 5, 6}:
                val_pos, title_pos, _, N_trim_pos = operate_mat(
                    mat_path, op, 1, analysis,
                    exceed_pct=exceed_pct,
                    bss_array=bss_all if op == 9 else None   # pass only when needed
                )
                val_neg, title_neg, _, N_trim_neg = operate_mat(
                    mat_path, op, 2, analysis,
                    exceed_pct=exceed_pct,
                    bss_array=bss_all if op == 9 else None
                )
        
            if analysis == 7:
                cfr, title_cfr, _, _ = operate_mat(
                    mat_path, op, 3, analysis,
                    exceed_pct=exceed_pct,
                    bss_array=bss_all if op == 9 else None
                )
        except Exception as err:
            # Log but keep the pipeline alive
            print(f"[process_file] {mat_path}: {err}")

        # ALWAYS return the 8-element tuple in the same order
        return (
            val_pos, title_pos,
            val_neg, title_neg,
            cfr,     title_cfr,
            N_trim_pos, N_trim_neg,
        )

    # Process files in batches to control memory usage.
    for i in range(0, len(um_files), batch_size):
        batch = um_files[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_worker) as executor:
            futures = {executor.submit(process_file, f): f for f in batch}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc=f'Processing Batch {i // batch_size + 1}'):
                fname = futures[future]
                try:
                    res = future.result()
                    (vp, tp, vn, tn, cfr_val, cfr_title, npos, nneg) = res

                    with lock:
                        if vp is not None and vn is not None:
                            val_pos_dict[fname] = vp
                            pos_title_dict[fname] = tp
                            val_neg_dict[fname] = vn
                            neg_title_dict[fname] = tn
                            N_trim_pos_dict[fname] = npos
                            N_trim_neg_dict[fname] = nneg

                        if cfr_val is not None:
                            cfr_dict[fname] = cfr_val
                            cfr_title_dict[fname] = cfr_title

                except Exception as e:
                    print(f"Error in future for {fname}: {e}")
        # Clear memory after each batch
        gc.collect()

    sorted_files = sorted(um_files, key=extract_sort_key)

    # Build lists in correct grain-size order
    val_pos_list = [val_pos_dict[f] for f in sorted_files if f in val_pos_dict]
    pos_title = [pos_title_dict[f]
                 for f in sorted_files if f in pos_title_dict]
    val_neg_list = [val_neg_dict[f] for f in sorted_files if f in val_neg_dict]
    neg_title = [neg_title_dict[f]
                 for f in sorted_files if f in neg_title_dict]
    cfr_list = [cfr_dict[f] for f in sorted_files if f in cfr_dict]
    cfr_title = [cfr_title_dict[f]
                 for f in sorted_files if f in cfr_title_dict]

    # N_trim arrays
    N_trim_pos_list = [N_trim_pos_dict[f]
                       for f in sorted_files if f in N_trim_pos_dict]
    N_trim_neg_list = [N_trim_neg_dict[f]
                       for f in sorted_files if f in N_trim_neg_dict]

    # ------------------------------------------------------------------------
    # Pre process for Analysis=1 (optional logic):
    # If analysis=1, you might separate depositional vs erosive, etc.
    # ------------------------------------------------------------------------

# %% Analysis

    if op != 6:
        N_trim_pos_list = None
        N_trim_neg_list = None

    # Now call process_flux_data
    dataset, dataset_title = process_flux_data(
        analysis=analysis,
        grain_sizes_w=grain_sizes_w,
        factor_sed_mass=factor_sed_mass,
        val_pos_list=val_pos_list,
        val_neg_list=val_neg_list,
        cfr_list=cfr_list,
        pos_title=pos_title,
        neg_title=neg_title,
        cfr_title=cfr_title,
        N_trim_pos_list=N_trim_pos_list,
        N_trim_neg_list=N_trim_neg_list,
        thresh=thresh
    )

    # Remove previous .xyz and .jpg files
    for xyz_file in glob.glob(os.path.join(main_path, "*.xyz")):
        os.remove(xyz_file)
    print("Deleted Previous XYZ Files")

    for jpg_file in glob.glob(os.path.join(main_path, "*.jpg")):
        os.remove(jpg_file)
    print("Deleted Previous JPG Files")

    # Save the analysis file
    with h5py.File(hyd_grid[0], 'r') as file:
        data = file['data']
        x = data['X'][:, :]
        y = data['Y'][:, :]

    # Write each grain size's dataset to text files
    if dataset is not None:
        for um in tqdm(range(len(dataset_title)), desc='Writing flux data to files',
                       total=len(dataset_title), leave=True, mininterval=0.1):

            val_um = dataset[:, :, um]
            title = dataset_title[um]

            # Ensure coordinate alignment
            if x.shape == val_um.shape:
                x_cor, y_cor = x, y
            else:
                x_cor, y_cor = x[1:, 1:], y[1:, 1:]

            x_flat = x_cor.flatten()
            y_flat = y_cor.flatten()
            z_flat = val_um.flatten()

            xyz = np.vstack((x_flat, y_flat, z_flat)).T
            header_str = "X Y Z"
            title_safe = safe_filename(title)
            file_title = os.path.join(main_path, f"{title_safe}.xyz")

            np.savetxt(file_title, xyz, fmt="%f %f %.12f", header=header_str)

        print("Flux data successfully written to .xyz files.")
    else:
        print("No dataset generated. Check if val_pos_list or val_neg_list was empty.")

    # Normalize and convert to relative percentages
    process_files = sorted(find_files(main_path, "*um.xyz"),
                           key=lambda x: int(re.search(r"(\d+)um", x).group(1)))

    if len(process_files) > 0:
        column_name = [re.search(r"(\d+)um", name).group(1)
                       for name in process_files]
        df = combine_xyz_files(process_files)
        df.columns = column_name
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Log-transform to reduce outliers
        # df = np.log1p(df)
        # Clip any negative rounding
        df = df.clip(lower=0)
        df_sum = df.sum(axis=1)
        df.loc[df_sum == 0, df.columns] = 1 / len(df.columns)

        # Normalize to sum=100
        df_sum = df.sum(axis=1)
        df = df.div(df_sum, axis=0) * 100

        # Correct rounding so sums ~100
        df['sum_diff'] = 100 - df.sum(axis=1)
        df = df.add(df['sum_diff'] / len(df.columns), axis=0)
        df.drop(columns=['sum_diff'], inplace=True)

        normalized_df = df

        # Write normalized data to .xyz
        xy_data = pd.read_csv(
            process_files[0], sep=' ', skiprows=1, header=None)
        xy_data.columns = ['x', 'y', 'z']
        xy_data = xy_data.drop(['z'], axis=1)

        for column in tqdm(normalized_df.columns, desc='Writing normalized % files',
                           total=len(normalized_df.columns), leave=True, mininterval=0.1):
            xyz_data = xy_data.copy()
            xyz_data['z'] = df[column]
            filename = f"{column}_perc_{model_type}.xyz"
            xyz_data.to_csv(os.path.join(main_path, filename),
                            sep=',', index=False, header=False, float_format="%.12f")

        print("Normalized flux data successfully written to .xyz files.")
    else:
        print("No '*um.xyz' files found. Skipping normalization step.")
# %% D50 map generation code

    print("\nD50 map generation")
    if d50_on == 0:
        print("D50 won't be calculated.")
    else:
        if '63' in df.columns:
            df = df.drop('63', axis=1)  # remove mud fraction if present

        d50_ranges = {
            1: ['105', '150', '210', '300', '420', '600'],
            2: ['150', '210', '300', '420'],
            3: ['105', '150', '210', '300', '420'],
            4: ['130', '160', '200', '255', '360'],
            5: ['130', '160', '200', '255', '360', '500']
        }

        if choice_d50 not in d50_ranges:
            raise ValueError(f"Invalid choice_d50={choice_d50}.")

        selected_grain_sizes = d50_ranges[choice_d50]
        grain_class = "-".join(selected_grain_sizes)
        valid_grain_sizes = [
            sz for sz in selected_grain_sizes if sz in df.columns]
        if not valid_grain_sizes:
            raise ValueError("No valid grain sizes found for D50 calculation!")

        df = df[valid_grain_sizes]
        df = sp_interp(df, valid_grain_sizes, 50)

        xy_data['z'] = df['D50']
        filename = f"D50_{model_type}_{grain_class}.xyz"
        xy_data.to_csv(os.path.join(main_path, filename),
                       sep=',', index=False, header=False)
        print(f"D50 data successfully written to {filename}.")

    # Convert to Bed Layer Fraction
    xyz_files = find_files(main_path, f"*perc_{model_type}.xyz")
    column_name = []
    for name in xyz_files:
        match_perc = re.search(
            r"(\d+)_perc_" + re.escape(model_type) + r".xyz", name)
        if match_perc:
            column_name.append(match_perc.group(1))

    if len(xyz_files) > 0:
        perc_df = combine_plot_files(xyz_files)
        perc_df.columns = column_name
        # Convert to bed layer fraction
        perc_df = perc_df / 100
        zero_sum_rows = perc_df.sum(axis=1) == 0
        perc_df.loc[zero_sum_rows, :] = 1 / len(perc_df.columns)
        perc_df = perc_df.div(perc_df.sum(axis=1), axis=0)
        perc_df.fillna(1 / len(perc_df.columns), inplace=True)

        xy_data = pd.read_csv(xyz_files[0], sep=',', skiprows=3, header=None)
        xy_data.columns = ['x', 'y', 'z']
        xy_data = xy_data.drop(['z'], axis=1)

        for col in tqdm(perc_df.columns, desc='Writing bed layer fraction files',
                        total=len(perc_df.columns), leave=True, mininterval=0.1):
            xyz_data = xy_data.copy()
            xyz_data['z'] = perc_df[col]
            filename = f"{col}_{model_type}_bl.xyz"
            xyz_data.to_csv(os.path.join(main_path, filename),
                            sep=',', index=False, header=False)

        print("Bed layer fraction data successfully written to .xyz files.")
    else:
        print(
            "No '*perc_{model_type}.xyz' files found. Skipping bed layer fraction step.")

    end_time = time.time()
    elapsed = end_time - st
    print(f'\n\nTotal time taken: {np.round((elapsed/60), 2)} mins')
