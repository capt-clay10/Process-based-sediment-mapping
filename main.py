'''
Accumulated Sediment Flux (ASF) — Sediment Distribution Generator
=================================================================

Generates sediment distribution maps from Delft3D-4 accumulated flux
outputs, following the ASF method introduced in:

    Soares et al. (under review), "Process-based sediment mapping — an
    alternative for dynamic and data-scarce coasts",
    Applied Ocean Research.

The ASF method estimates a quasi-equilibrium relative seabed sediment
distribution by computing frequency-weighted gross bed-material
exchange from modelled accumulated flux time series (Equations 1–4 in
the manuscript).  A user-defined trimming proportion (α) symmetrically
removes infrequent flux values before averaging, so that frequent
transport events govern the predicted distribution.

This script also contains an optional trimming-proportion calibration
workflow that sweeps α over a user-defined range, evaluates the
distributional fit (W1norm) against grab-sample data, and recommends
an optimal α using several complementary criteria.

Experimental (non-ASF) flux descriptors and analysis combinations are
retained in a separate section for research purposes.

Required inputs
---------------
- Per-grain-size accumulated flux files exported from Delft3D-4 as
  v7.3 / HDF5 MAT files, named {grain_size}um_{model_type}.mat
  (e.g. 105um_sf.mat).
- A hydrodynamic-grid MAT file (single time step, k-layer = 1).

Optional inputs
---------------
- Grab-sample validation CSV(s) for trimming calibration.
- Bed-shear-stress MAT file for experimental operations 7–8.

Expected outputs
----------------
- *um.xyz          raw flux per grain size
- *_perc_*.xyz     normalised percentage availability
- *_bl.xyz         bed-layer fractions (sum = 1)
- D50_*.xyz        median grain-size map
- Trimming calibration report + figure (if enabled)

Grain sizes are user-defined and not restricted to a fixed set.
Phi-scale intervals are generated automatically from user-specified
bounds and interval width.

Author : Clayton Soares
'''
# %% Section 1: Import packages
from __future__ import annotations
from threading import Lock
import concurrent.futures
import gc
import glob
import h5py
import hashlib
import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import psutil
import re
import scipy.stats as ss
import time
import warnings
from datetime import datetime
from itertools import combinations
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# %% Section 2: User parameters

# Modify the values below to match your project.  Every path that is not
# needed for a particular run can be set to None.


# ---- Section 2.1: Requested tasks --------------------------------------

calib_trim   = True           # True  → run trimming calibration before ASF
asf_only     = True           # True  → run only the ASF method (op=6, an=6)
                              # False → run the experimental section as well
generate_d50 = True           # True  → produce a D50 map at the end

# ---- Section 2.2: Calibration ------------------------------------------
# Validation files should be CSV with columns:
#   X, Y, um_{gs1}, um_{gs2}, ...
# where {gsN} is the representative grain size in microns for each
# phi interval (e.g. um_105, um_150, ...).  Each um_* column holds the
# distributional availability (proportion or %) for that fraction at
# that sample point.  The script auto-detects the format and
# renormalises to proportions (sum = 1).
#
# If you have separate files per zone, list them all in calib_trim_units
# and set calib_trim_one_unit = False.  Each filename (minus extension)
# becomes a zone key, and a pooled 'all' zone is added automatically.
# If you have a single combined file, set calib_trim_one_unit = True.

calib_trim_one_unit = False
input_grab_sample_location = r'C:\Final_models\paper_2_sed_flux\Revision\Sim4_ASF_PWI\Grab_sample_data'
calib_trim_units = ["offshore_grab_samples.csv", "channel_grab_samples.csv", "flats_grab_samples.csv"]

trim_start = 0               # starting trim % (from each tail)
trim_end   = 35              # ending trim %
trim_step  = 1               # step size

w1norm_mode = 'full'          # 'percentile' → D10, D25, D50, D75, D90 
                              # 'full'       → all phi-bin fractions

# ---- Section 2.3: Directories ------------------------------------------

input_mat_folder           = r'E:\delft3d_python\Model_based_methods\NEWS'
                              # folder containing *um_{model_type}.mat files
input_grid_mat             = r'E:\delft3d_python\Model_based_methods\NEWS\news_hyd_grid.mat'
                              # hydrodynamic grid file (can be in a different folder)
output_ASF_folder          = r'E:\delft3d_python\Model_based_methods\Revision_paper2\news_test'
                              # final output .xyz files go here
output_ASF_interim_folder  = r'E:\delft3d_python\Model_based_methods\Revision_paper2\news_test'
                              # per-grain cached .npz files go here
output_trim_calib_folder   = r'E:\delft3d_python\Model_based_methods\Revision_paper2\news_test'
                              # calibration report + figures saved here
output_trim_interim_folder = r'E:\delft3d_python\Model_based_methods\Revision_paper2\news_test'
                              # calibration cached .npy / .csv files go here
input_bss_mat              = None
                              # bed-shear-stress MAT file (only for op 7–8)

# ---- Section 2.4: ASF parameters ---------------------------------------

max_workers = 4               # number of parallel threads for grain-size processing
batch_size  = 2               # grain-size files processed per batch
trim_alpha  = 25              # default trimming proportion (%) if calibration is off
grain_sizes = [105, 150, 210, 300, 420]
                              # grain sizes in µm — must match the MAT filenames

# Phi-scale settings — the code builds phi intervals from these values.
# finest_phi is the upper phi bound (finest grain), coarsest_phi is the
# lower bound (coarsest grain), and phi_interval is the bin width.
# The number of intervals must equal the number of grain sizes.
finest_phi   = 3.5
coarsest_phi = 1.0
phi_interval = 0.5

# Critical-shear-stress weighting — method and factor are separate.
# method selects the formulation; factor scales the normalised weights.
# When method = 0 OR factor = 0, no weighting is applied (all weights = 1).
shear_weight_method = 0       # 0 = no weighting, 1 = Van Rijn (1984), 2 = Soulsby (1997)
shear_weight_factor = 1.0     # scaling multiplier (only used when method ≠ 0)

# ---- Section 2.5: Input model parameters --------------------------------

model_type = 'news'           # suffix in MAT filenames, e.g. 105um_news.mat

# %% ---- Section 2.6: Experimental approaches ----------------------------
# These settings are only used when asf_only = False.
# ASF always uses trimmed-mean + frequency-weighted total transport
# (op = 6, analysis = 6 internally).

operation_on_fluxes = 1
    # 1 = arithmetic mean       5 = median
    # 2 = 95th percentile       6 = 75th percentile
    # 3 = minimum               7 = exceedance mean
    # 4 = maximum               8 = BSS-conditioned mean

exceedance_pctile = 25.0      # percentile threshold for operations 7 and 8

analysis_on_operated_fluxes = 1
    # 1 = combined flux (split by grain_size_thresh)
    # 2 = positive flux only    4 = residual depositional
    # 3 = negative flux only    5 = geometric mean
    #                           6 = cumulative flux ratio (CFR)

grain_size_thresh = 300       # grain-size threshold for analysis 1 (µm)
# for analysis 1 the threshold dictates after which grain size from your list
# only the negative (erosive) fluxes will be taken and before that only 
# positive (depositive) fluxes will be taken


# %% Section 3: Utility functions
# =========================================================================

def extract_sort_key(filename):
    match = re.search(r"(\d+)um", filename)
    return int(match.group(1)) if match else filename

def find_files(root_dir, file_extension):
    matches = []
    for dirpath, _, _ in os.walk(root_dir):
        for f in glob.glob(os.path.join(dirpath, file_extension)):
            matches.append(f)
    return sorted(matches, key=extract_sort_key)

INVALID_CHARS = r'<>:"/\|?*'

def safe_filename(s: str, repl: str = "_") -> str:
    return re.sub(f"[{re.escape(INVALID_CHARS)}]", repl, s.strip())

def ensure_dir(path: str) -> None:
    if path is not None:
        os.makedirs(path, exist_ok=True)

def combine_tabular_files(files, sep=' ', skiprows=1):
    combined = []
    for f in files:
        data = pd.read_csv(f, sep=sep, skiprows=skiprows, header=None)
        combined.append(data.iloc[:, -1])
    return pd.concat(combined, axis=1)


def align_grid_to_data(x_grid, y_grid, data_shape):
    """Align Delft3D grid coordinates to the flux-data shape.

    Delft3D stores node (corner) coordinates on a staggered grid,
    while accumulated-flux data lives at cell centres.  The physically
    correct cell-centre coordinate is the average of its four
    surrounding corners:

        x_centre(i,j) = mean(x_corner(i:i+2, j:j+2))

    This function handles all Delft3D grid/data combinations:
      - Grid == data shape  → already cell-centred, return as-is
      - Grid larger in one or both dims → compute cell centres by
        averaging adjacent nodes along the oversized dimension(s)

    Parameters
    ----------
    x_grid, y_grid : ndarray (gx, gy)
        Node/corner coordinates from the hydrodynamic grid file.
    data_shape : tuple (dx, dy)
        The 2-D shape of the flux data arrays.

    Returns
    -------
    x_c, y_c : ndarray (dx, dy)
        Coordinates aligned to the data grid.
    """
    gx, gy = x_grid.shape
    dx, dy = data_shape

    if (gx, gy) == (dx, dy):
        return x_grid, y_grid

    # compute cell centres by averaging adjacent nodes along
    # whichever dimensions are oversized
    x_c, y_c = x_grid.copy(), y_grid.copy()

    if gx == dx + 1 and gy == dy + 1:
        # standard Delft3D: corners (M+1, N+1) → centres (M, N)
        x_c = 0.25 * (x_grid[:-1, :-1] + x_grid[1:, :-1]
                       + x_grid[:-1, 1:] + x_grid[1:, 1:])
        y_c = 0.25 * (y_grid[:-1, :-1] + y_grid[1:, :-1]
                       + y_grid[:-1, 1:] + y_grid[1:, 1:])
    elif gx == dx + 1 and gy == dy:
        # oversized in dim-0 only: average adjacent rows
        x_c = 0.5 * (x_grid[:-1, :] + x_grid[1:, :])
        y_c = 0.5 * (y_grid[:-1, :] + y_grid[1:, :])
    elif gx == dx and gy == dy + 1:
        # oversized in dim-1 only: average adjacent columns
        x_c = 0.5 * (x_grid[:, :-1] + x_grid[:, 1:])
        y_c = 0.5 * (y_grid[:, :-1] + y_grid[:, 1:])
    else:
        raise ValueError(
            f"Cannot align grid shape {x_grid.shape} to data shape "
            f"{data_shape}. Expected grid to be equal or +1 in each "
            f"dimension.  Check that the grid file matches the flux files.")

    return x_c, y_c


def load_grid_coordinates(grid_file):
    """Load X and Y coordinate arrays from a Delft3D hydrodynamic grid file.

    Handles both 2-D grids (shape M×N) and 3-D grids with a leading
    k-layer dimension (shape K×M×N), as produced by different Delft3D
    export configurations.  When 3-D, the first k-layer (typically the
    bed layer) is used.

    Parameters
    ----------
    grid_file : str
        Path to the HDF5 / v7.3 MAT grid file.

    Returns
    -------
    x, y : ndarray (M, N)
    """
    with h5py.File(grid_file, 'r') as f:
        x_raw = f['data/X'][:]
        y_raw = f['data/Y'][:]

    # if 3-D (k-layers × M × N), take the first layer
    if x_raw.ndim == 3:
        x_raw = x_raw[0, :, :]
    if y_raw.ndim == 3:
        y_raw = y_raw[0, :, :]

    return x_raw, y_raw


# %% Section 4: Phi-scale and grain-size functions
# =========================================================================

def generate_phi_intervals(finest_phi, coarsest_phi, phi_step):
    intervals = []
    phi = finest_phi
    while phi - phi_step >= coarsest_phi - 1e-9:
        intervals.append((round(phi, 4), round(phi - phi_step, 4)))
        phi -= phi_step
    return intervals

def map_grains_to_phi(grain_sizes_um, phi_intervals):
    mapping = {}
    phi_values = -np.log2(np.array(grain_sizes_um, dtype=float) / 1000.0)
    for gs, phi_val in zip(grain_sizes_um, phi_values):
        for (phi_upper, phi_lower) in phi_intervals:
            if phi_upper >= phi_val > phi_lower:
                mapping[(phi_upper, phi_lower)] = gs
                break
        else:
            if abs(phi_val - phi_intervals[0][0]) < 1e-6:
                mapping[phi_intervals[0]] = gs
    smallest_gs = min(grain_sizes_um)
    if phi_intervals[0] not in mapping:
        mapping[phi_intervals[0]] = smallest_gs
    return mapping

def sp_interp(df_in, grain_sizes_um, percentile,
              phi_intervals=None, n_fine=60):
    if phi_intervals is None:
        phi_intervals = generate_phi_intervals(finest_phi, coarsest_phi,
                                               phi_interval)
    grain_to_phi = map_grains_to_phi(grain_sizes_um, phi_intervals)
    overall_upper = max(p[0] for p in phi_intervals)
    overall_lower = min(p[1] for p in phi_intervals)
    fine_phi = np.linspace(overall_upper, overall_lower, n_fine)
    df = df_in.copy()
    col_name = f'D{int(percentile)}'
    d_values = []
    for _, row in df.iterrows():
        fine_dist = np.zeros_like(fine_phi)
        for (phi_upper, phi_lower), gs in grain_to_phi.items():
            gs_str = str(int(gs))
            if gs_str not in row.index:
                continue
            avail = row[gs_str]
            mask = (fine_phi >= phi_lower) & (fine_phi <= phi_upper)
            n_pts = np.sum(mask)
            if n_pts > 0:
                fine_dist[mask] += avail / n_pts
        cum = np.cumsum(fine_dist)
        if cum[-1] > 0:
            cum_pct = (cum / cum[-1]) * 100.0
        else:
            cum_pct = np.zeros_like(cum)
        try:
            if len(np.unique(cum_pct)) > 1:
                d_phi = np.interp(percentile, cum_pct, fine_phi)
                d_um = 2 ** (-d_phi) * 1000.0
            else:
                d_um = np.nan
        except ValueError:
            d_um = np.nan
        d_values.append(round(d_um, 2))
    df[col_name] = d_values
    return df

def sp_interp_row(fractions, grain_sizes_um, percentile,
                  phi_intervals=None, n_fine=60):
    if phi_intervals is None:
        phi_intervals = generate_phi_intervals(finest_phi, coarsest_phi,
                                               phi_interval)
    grain_to_phi = map_grains_to_phi(grain_sizes_um, phi_intervals)
    overall_upper = max(p[0] for p in phi_intervals)
    overall_lower = min(p[1] for p in phi_intervals)
    fine_phi = np.linspace(overall_upper, overall_lower, n_fine)
    fine_dist = np.zeros_like(fine_phi)
    gs_list = [int(gs) for gs in grain_sizes_um]
    for (phi_upper, phi_lower), gs in grain_to_phi.items():
        idx = gs_list.index(int(gs)) if int(gs) in gs_list else None
        if idx is None or idx >= len(fractions):
            continue
        avail = fractions[idx]
        mask = (fine_phi >= phi_lower) & (fine_phi <= phi_upper)
        n_pts = np.sum(mask)
        if n_pts > 0:
            fine_dist[mask] += avail / n_pts
    cum = np.cumsum(fine_dist)
    if cum[-1] > 0:
        cum_pct = (cum / cum[-1]) * 100.0
    else:
        return np.nan
    try:
        if len(np.unique(cum_pct)) > 1:
            d_phi = np.interp(percentile, cum_pct, fine_phi)
            return round(2 ** (-d_phi) * 1000.0, 2)
        else:
            return np.nan
    except ValueError:
        return np.nan


# %% Section 5: Critical shear-stress weighting
# =========================================================================

def compute_tau_c_soulsby(grain_sizes_m,
                          rho_s=2650, rho=1000, g=9.81, nu=1e-6):
    s = rho_s / rho
    d_star = grain_sizes_m * ((g * (s - 1)) / (nu**2)) ** (1/3)
    theta_c = (0.30 * (1 + 1.2 * d_star) ** (-1)
               + 0.055 * (1 - np.exp(-0.020 * d_star)))
    return theta_c * (rho_s - rho) * g * grain_sizes_m

def compute_tau_c_vanrijn(grain_sizes_m,
                          rho_s=2650, rho=1000, g=9.81, nu=1e-6):
    s = rho_s / rho
    D_star = grain_sizes_m * ((g * (s - 1)) / (nu**2)) ** (1/3)
    theta_c = 0.24 * (D_star ** -1.0) + 0.055 * (D_star ** -0.6)
    return theta_c * (rho_s - rho) * g * grain_sizes_m

def compute_shear_weights(grain_sizes_um, method, factor):
    if factor == 0 or method == 0:
        return np.ones(len(grain_sizes_um))
    gs_m = np.array(grain_sizes_um, dtype=float) * 1e-6
    if method == 1:
        tau = compute_tau_c_vanrijn(gs_m)
    else:
        tau = compute_tau_c_soulsby(gs_m)
    tau_norm = tau / tau.max()
    return tau_norm * factor


# %% Section 6: Core flux computation
# =========================================================================

def compute_trimmed_mean_flux(file_name, trim_pct):
    """Equations 2 and 3 from the manuscript."""
    gs_match = re.search(r"(\d+)um", file_name)
    grain_size = gs_match.group(1) if gs_match else "unknown"
    with h5py.File(file_name, 'r') as f:
        val_real = f['data/Val'][:]
    val_c = val_real.astype(np.float32, copy=True)

    val_pos_raw = np.where(val_c > 0, val_c, np.nan).astype(float)
    val_pos_raw *= 1000.0
    lower = np.nanpercentile(val_pos_raw, trim_pct, axis=2, keepdims=True)
    upper = np.nanpercentile(val_pos_raw, 100 - trim_pct, axis=2, keepdims=True)
    mask_pos = (val_pos_raw >= lower) & (val_pos_raw <= upper)
    with np.errstate(invalid='ignore'):
        val_pos = np.nanmean(np.where(mask_pos, val_pos_raw, np.nan), axis=2)
    N_trim_pos = np.sum(mask_pos & ~np.isnan(val_pos_raw), axis=2)

    val_neg_raw = np.where(val_c < 0, np.abs(val_c), np.nan).astype(float)
    val_neg_raw *= 1000.0
    lower = np.nanpercentile(val_neg_raw, trim_pct, axis=2, keepdims=True)
    upper = np.nanpercentile(val_neg_raw, 100 - trim_pct, axis=2, keepdims=True)
    mask_neg = (val_neg_raw >= lower) & (val_neg_raw <= upper)
    with np.errstate(invalid='ignore'):
        val_neg = np.nanmean(np.where(mask_neg, val_neg_raw, np.nan), axis=2)
    N_trim_neg = np.sum(mask_neg & ~np.isnan(val_neg_raw), axis=2)

    val_pos = np.nan_to_num(val_pos, nan=0.0) / 1000.0
    val_neg = np.nan_to_num(val_neg, nan=0.0) / 1000.0
    return val_pos, val_neg, N_trim_pos, N_trim_neg, grain_size

def assemble_asf(val_pos_list, val_neg_list,
                 N_trim_pos_list, N_trim_neg_list,
                 grain_sizes_um, weight_method, weight_factor):
    """Equation 4 from the manuscript."""
    pos_all = np.stack(val_pos_list, axis=2)
    neg_all = np.stack(val_neg_list, axis=2)
    N_pos_all = np.stack(N_trim_pos_list, axis=2).astype(float)
    N_neg_all = np.stack(N_trim_neg_list, axis=2).astype(float)
    N_total = N_pos_all + N_neg_all
    N_total[N_total == 0] = 1.0
    frac_pos = N_pos_all / N_total
    frac_neg = N_neg_all / N_total
    dataset = frac_pos * pos_all + frac_neg * neg_all
    weights = compute_shear_weights(grain_sizes_um, weight_method,
                                    weight_factor)
    dataset *= weights.reshape(1, 1, -1)
    return dataset


# %% Section 6b: Experimental flux descriptors
# =========================================================================

def operate_mat(file_name, operation, choice, analysis, *,
                exceed_pct=25.0, bss_array=None):
    gs_match = re.search(r"(\d+)um", file_name)
    grain_size = int(gs_match.group(1)) if gs_match else "unknown"
    with h5py.File(file_name, "r") as f:
        dset = f["data"]
        units = "".join(chr(i) for i in dset["Units"][:].ravel())
        name = "".join(chr(i) for i in dset["Name"][:].ravel())
        title_mid_end = "".join(
            c for c in name if c.isalpha() or c.isspace()).lower()
        val_real = dset["Val"][:]
    BYTES_PER_GB = 1024 ** 3
    avail_gb = psutil.virtual_memory().available / BYTES_PER_GB
    req_gb = val_real.nbytes / BYTES_PER_GB
    dtype = np.float32 if req_gb > 0.9 * avail_gb else val_real.dtype
    val_c = val_real.astype(dtype, copy=True)
    if choice == 1:
        val = np.where(val_c > 0, val_c, np.nan).astype(float, copy=False)
        title_mid = "positive"
    elif choice == 2:
        val = np.where(val_c < 0, np.abs(val_c), np.nan).astype(float, copy=False)
        title_mid = "negative"
    else:
        val = val_c.astype(float, copy=True)
        title_mid = "total"
    val *= 1000.0

    def _exceedance_mean(arr):
        thr = np.nanpercentile(arr, exceed_pct, axis=2, keepdims=True)
        msk = arr >= thr
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.where(msk, arr, np.nan), axis=2)

    if analysis == 6:
        pos_flux = np.where(val_c > 0, val_c, np.nan) * 1000.0
        neg_flux = np.where(val_c < 0, np.abs(val_c), np.nan) * 1000.0
        ops = {
            1: (np.nanmean, "Mean CFR"),
            2: (lambda a, ax: np.nanpercentile(a, 95, ax), "95th Perc CFR"),
            3: (np.nanmin, "Minimum CFR"),
            4: (np.nanmax, "Maximum CFR"),
            5: (np.nanmedian, "Median CFR"),
            6: (lambda a, ax: np.nanpercentile(a, 75, ax), "75th Perc CFR"),
        }
        if operation in ops:
            fn, title_start = ops[operation]
            pos_stat = fn(pos_flux, 2)
            neg_stat = fn(neg_flux, 2)
        elif operation == 7:
            pos_stat = np.nan_to_num(_exceedance_mean(pos_flux), nan=0.0)
            neg_stat = np.nan_to_num(_exceedance_mean(neg_flux), nan=0.0)
            title_start = f"CFR Mean > {exceed_pct:g}th Perc"
        elif operation == 8:
            if bss_array is None:
                raise ValueError("bss_array required for operation 8")
            bss_thr = np.nanpercentile(bss_array, exceed_pct, axis=2, keepdims=True)
            mask = bss_array >= bss_thr
            with np.errstate(invalid="ignore"):
                pos_stat = np.nanmean(np.where(mask, pos_flux, np.nan), axis=2)
                neg_stat = np.nanmean(np.where(mask, neg_flux, np.nan), axis=2)
            pos_stat = np.nan_to_num(pos_stat, nan=0.0)
            neg_stat = np.nan_to_num(neg_stat, nan=0.0)
            title_start = f"CFR | BSS>{exceed_pct:g}th Perc"
        else:
            raise ValueError("operation must be 1–8 for CFR analysis")
        neg_stat = np.where(neg_stat < 1e-12, 1e-12, neg_stat)
        val_in = np.maximum(pos_stat / neg_stat, 1) - 1
        title_mid = "total"
        return val_in, f"{title_start} {title_mid} {title_mid_end} {grain_size}um", units

    if   operation == 1: val_in, title_start = np.nanmean(val, axis=2), "Mean"
    elif operation == 2: val_in, title_start = np.nanpercentile(val, 95, axis=2), "P95"
    elif operation == 3: val_in, title_start = np.nanmin(val, axis=2), "Minimum"
    elif operation == 4: val_in, title_start = np.nanmax(val, axis=2), "Maximum"
    elif operation == 5: val_in, title_start = np.nanmedian(val, axis=2), "Median"
    elif operation == 6: val_in, title_start = np.nanpercentile(val, 75, axis=2), "P75"
    elif operation == 7:
        val_in = np.nan_to_num(_exceedance_mean(val), nan=0.0)
        title_start = f"Mean > {exceed_pct:g}th Perc"
    elif operation == 8:
        if bss_array is None: raise ValueError("bss_array required for operation 8")
        if bss_array.shape[2] != val_c.shape[2]: raise ValueError("Time-dimension mismatch")
        bss_thr = np.nanpercentile(bss_array, exceed_pct, axis=2, keepdims=True)
        mask = bss_array >= bss_thr
        with np.errstate(invalid="ignore"):
            val_in = np.nanmean(np.where(mask, val, np.nan), axis=2)
        val_in = np.nan_to_num(val_in, nan=0.0)
        title_start = f"Mean | BSS>{exceed_pct:g}th Perc"
    else:
        raise ValueError("operation must be 1–8")
    val_in /= 1000.0
    return val_in, f"{title_start} {title_mid} {title_mid_end} {grain_size}um", units

def process_experimental_flux(analysis, val_pos_list, val_neg_list,
                              cfr_list, pos_title, neg_title, cfr_title,
                              grain_sizes_um, weight_method, weight_factor,
                              thresh=None):
    weights = compute_shear_weights(grain_sizes_um, weight_method, weight_factor)
    w = weights.reshape(1, 1, -1)
    pos_all = np.stack(val_pos_list, axis=2) if len(val_pos_list) > 0 else None
    neg_all = np.stack(val_neg_list, axis=2) if len(val_neg_list) > 0 else None
    cfr_all = np.stack(cfr_list, axis=2) if len(cfr_list) > 0 else None
    dataset = None
    dataset_title = []
    if analysis == 1:
        if pos_all is None or neg_all is None: return None, []
        split_vals, split_titles = [], []
        for i, title in enumerate(pos_title):
            m = re.search(r'(\d+)um\b', title)
            if not m: continue
            gs = int(m.group(1))
            if gs <= thresh:
                split_vals.append(pos_all[:, :, i])
                split_titles.append(title.replace("positive", "Deposition"))
            else:
                split_vals.append(neg_all[:, :, i])
                split_titles.append(neg_title[i].replace("negative", "Erosion"))
        if not split_vals: return None, []
        dataset = np.stack(split_vals, axis=2)
        dataset_title = split_titles
    elif analysis == 2: dataset, dataset_title = pos_all, pos_title
    elif analysis == 3: dataset, dataset_title = neg_all, neg_title
    elif analysis == 4:
        if pos_all is not None and neg_all is not None:
            d = pos_all - neg_all; d[d < 0] = 0.0; d[np.abs(d) < 1e-6] = 0.0
            dataset = d
            dataset_title = [s.replace("positive", "Residual Depositional") for s in pos_title]
    elif analysis == 5:
        if pos_all is not None and neg_all is not None:
            dataset = np.sqrt(np.maximum(pos_all, 0.0) * np.maximum(neg_all, 0.0) + 1e-12)
            dataset_title = [s.replace("negative", "Geometric Mean") for s in neg_title]
    elif analysis == 6:
        if cfr_all is not None: dataset, dataset_title = cfr_all, cfr_title
    else: raise ValueError("analysis must be 1–6")
    if dataset is None: return None, []
    dataset = dataset * w
    return dataset, dataset_title


# %% Section 7: W1norm computation
# =========================================================================

def _um2phi(um):
    return -np.log2(np.asarray(um, dtype=float) / 1000.0)

def compute_point_w1norm_percentile(obs_pct, mod_pct):
    cols = ['D10', 'D25', 'D50', 'D75', 'D90']
    obs_um = np.array([obs_pct[c] for c in cols], dtype=float)
    mod_um = np.array([mod_pct[c] for c in cols], dtype=float)
    if np.any(np.isnan(obs_um)) or np.any(np.isnan(mod_um)):
        return np.nan
    obs_phi = _um2phi(obs_um)
    mod_phi = _um2phi(mod_um)
    iqr = np.nanpercentile(obs_phi, 75) - np.nanpercentile(obs_phi, 25)
    if iqr <= 0 or np.isnan(iqr):
        return np.nan
    return wasserstein_distance(obs_phi, mod_phi) / iqr

def compute_point_w1norm_full(obs_fractions, mod_fractions, phi_mids):
    obs = np.asarray(obs_fractions, dtype=float)
    mod = np.asarray(mod_fractions, dtype=float)
    if np.nansum(obs) == 0 or np.nansum(mod) == 0:
        return np.nan
    obs = obs / np.nansum(obs)
    mod = mod / np.nansum(mod)
    iqr = (np.nanpercentile(np.repeat(phi_mids, (obs * 1000).astype(int)), 75)
           - np.nanpercentile(np.repeat(phi_mids, (obs * 1000).astype(int)), 25))
    if iqr <= 0 or np.isnan(iqr):
        return np.nan
    return wasserstein_distance(phi_mids, phi_mids, obs, mod) / iqr


# %% Section 8: Validation data loading and W1norm calculation
# =========================================================================

def load_validation_data(grab_path, filenames, grain_sizes_um,
                         one_unit=False, phi_ivls=None):
    if phi_ivls is None:
        phi_ivls = generate_phi_intervals(finest_phi, coarsest_phi, phi_interval)
    gs_cols = [f'um_{int(gs)}' for gs in grain_sizes_um]
    str_cols = [str(int(gs)) for gs in grain_sizes_um]
    zonal = {}
    all_frames = []
    for fname in filenames:
        fpath = os.path.join(grab_path, fname)
        if not os.path.exists(fpath):
            print(f"    Warning: {fpath} not found, skipping")
            continue
        df = pd.read_csv(fpath)
        if 'X' in df.columns:
            df = df.rename(columns={'X': 'x_obs', 'Y': 'y_obs'})
        elif 'x' in df.columns:
            df = df.rename(columns={'x': 'x_obs', 'y': 'y_obs'})

        # --- grain-size column subsetting and renormalisation ---
        csv_um_cols = [c for c in df.columns if re.match(r'^um_\d+$', c)]
        matched = [c for c in gs_cols if c in csv_um_cols]
        dropped = [c for c in csv_um_cols if c not in gs_cols]

        print(f"    {fname}:")
        print(f"      CSV grain-size columns: {csv_um_cols}")
        print(f"      Matched to user list:   {matched}")
        if dropped:
            print(f"      DROPPED (not in grain_sizes): {dropped}")

        # rename matched columns
        rename_map = {}
        for gc, sc in zip(gs_cols, str_cols):
            if gc in df.columns:
                rename_map[gc] = sc
        df = df.rename(columns=rename_map)
        df = df.dropna(subset=['x_obs', 'y_obs'])

        # auto-detect format and normalise to proportions (0–1)
        present_str = [sc for gc, sc in zip(gs_cols, str_cols)
                       if gc in rename_map or sc in df.columns]
        if len(present_str) > 0:
            sub = df[present_str].astype(float)
            raw_median_sum = sub.sum(axis=1).median()

            if raw_median_sum > 10:
                fmt_detected = 'percentage (0–100)'
            else:
                fmt_detected = 'proportion (0–1)'
            print(f"      Input format detected:  {fmt_detected}"
                  f"  (median row sum = {raw_median_sum:.2f})")
            if dropped:
                print(f"      Subset median sum after dropping: {raw_median_sum:.4f}")

            # normalise: divide by row sum → always proportions (0–1)
            # this handles both percentage and proportion inputs,
            # and compensates for dropped fractions
            row_sum = sub.sum(axis=1)
            row_sum = row_sum.replace(0, 1.0)
            df[present_str] = sub.div(row_sum, axis=0)
            print(f"      Standardised to proportions (sum = 1.0 per row).")

        for pctile in [10, 25, 50, 75, 90]:
            vals = []
            for _, row in df.iterrows():
                fracs = row[[c for c in str_cols if c in df.columns]].values.astype(float)
                vals.append(sp_interp_row(fracs, grain_sizes_um, pctile, phi_ivls))
            df[f'D{pctile}_obs'] = vals
        zone_key = os.path.splitext(fname)[0]
        all_frames.append((zone_key, df))

    # always keep per-zone data
    for key, df in all_frames:
        zonal[key] = df

    # when one_unit=True OR multiple files provided, also add a pooled 'all'
    if one_unit or len(all_frames) > 1:
        combined = pd.concat([df for _, df in all_frames], ignore_index=True)
        zonal['all'] = combined

    return zonal

def compute_zonal_w1norm(model_df, validation_df, grain_sizes_um,
                         mode='percentile', phi_ivls=None):
    if phi_ivls is None:
        phi_ivls = generate_phi_intervals(finest_phi, coarsest_phi, phi_interval)
    str_cols = [str(int(gs)) for gs in grain_sizes_um]
    mdf = model_df.dropna(subset=['x', 'y']).copy()
    mdf = mdf[np.isfinite(mdf['x']) & np.isfinite(mdf['y'])]
    tree = cKDTree(mdf[['x', 'y']].values)
    val_coords = validation_df[['x_obs', 'y_obs']].values
    _, indices = tree.query(val_coords)
    w1_vals = []
    if mode == 'percentile':
        for i, idx in enumerate(indices):
            fracs = mdf.iloc[idx][[c for c in str_cols if c in mdf.columns]].values.astype(float)
            mod_pct = {}
            for p in [10, 25, 50, 75, 90]:
                mod_pct[f'D{p}'] = sp_interp_row(fracs, grain_sizes_um, p, phi_ivls)
            obs_pct = {}
            row = validation_df.iloc[i]
            for p in [10, 25, 50, 75, 90]:
                obs_pct[f'D{p}'] = row[f'D{p}_obs']
            w1_vals.append(compute_point_w1norm_percentile(obs_pct, mod_pct))
    else:
        phi_mids = np.array([0.5 * (u + l) for u, l in phi_ivls])
        for i, idx in enumerate(indices):
            mod_fracs = mdf.iloc[idx][[c for c in str_cols if c in mdf.columns]].values.astype(float)
            obs_fracs = validation_df.iloc[i][[c for c in str_cols if c in validation_df.columns]].values.astype(float)
            w1_vals.append(compute_point_w1norm_full(obs_fracs, mod_fracs, phi_mids))
    return np.array(w1_vals)


# %% Section 8b: ASF distribution generation for calibration
# =========================================================================

def generate_asf_distribution_fresh(mat_path, trim_pct, model_t,
                                    grain_sizes_um, wt_method, wt_factor,
                                    grid_mat_path=None):
    """Compute a full ASF distribution from scratch (no caching).
    Returns (DataFrame, x_coords, y_coords)."""
    um_files = find_files(mat_path, f"*um_{model_t}.mat")
    if not um_files:
        raise FileNotFoundError(f"No *um_{model_t}.mat files in {mat_path}")
    # resolve grid file: use explicit path if given, else search mat folder
    if grid_mat_path and os.path.isfile(grid_mat_path):
        grid_file = grid_mat_path
    else:
        hyd_grid = find_files(mat_path, 'hyd_grid_nf.mat')
        if not hyd_grid:
            raise FileNotFoundError(
                f"Hydrodynamic grid file not found in {mat_path}.\n"
                f"Set input_grid_mat to the correct path.")
        grid_file = hyd_grid[0]
    x, y = load_grid_coordinates(grid_file)
    val_pos_list, val_neg_list, N_pos_list, N_neg_list = [], [], [], []
    for mf in tqdm(um_files, desc=f"    Loading MAT files (trim={trim_pct}%)",
                   leave=False, ncols=80):
        vp, vn, np_, nn_, _ = compute_trimmed_mean_flux(mf, trim_pct)
        val_pos_list.append(vp); val_neg_list.append(vn)
        N_pos_list.append(np_); N_neg_list.append(nn_)
    dataset = assemble_asf(val_pos_list, val_neg_list,
                           N_pos_list, N_neg_list,
                           grain_sizes_um, wt_method, wt_factor)
    ref_shape = val_pos_list[0].shape
    x_c, y_c = align_grid_to_data(x, y, ref_shape)
    str_cols = [str(int(gs)) for gs in grain_sizes_um]
    rows = {'x': x_c.flatten(), 'y': y_c.flatten()}
    for k, gs_str in enumerate(str_cols):
        rows[gs_str] = dataset[:, :, k].flatten()
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
    gs_df = df[str_cols]
    row_sum = gs_df.sum(axis=1)
    gs_df.loc[row_sum == 0, :] = 1.0 / len(str_cols)
    row_sum = gs_df.sum(axis=1)
    gs_df = gs_df.div(row_sum, axis=0) * 100.0
    df[str_cols] = gs_df
    return df, x_c, y_c


# %% Section 9: Trimming calibration — selection criteria
# =========================================================================

def compute_pooled_median(results, trim_values):
    medians = []
    for tv in trim_values:
        pool = []
        for zone in results:
            if tv in results[zone]:
                v = results[zone][tv]; v = v[~np.isnan(v)]
                pool.extend(v.tolist())
        medians.append(np.median(pool) if pool else np.nan)
    return medians, trim_values[int(np.nanargmin(medians))]

def compute_per_zone_minima(results, trim_values):
    zone_opt = {}
    for zone in results:
        meds = []
        for tv in trim_values:
            if tv in results[zone]:
                v = results[zone][tv]; v = v[~np.isnan(v)]
                meds.append(np.median(v) if len(v) > 0 else np.nan)
            else: meds.append(np.nan)
        best = int(np.nanargmin(meds))
        zone_opt[zone] = (trim_values[best], meds[best])
    return zone_opt

def compute_kneedle(results, trim_values):
    pooled, _ = compute_pooled_median(results, trim_values)
    pooled = np.array(pooled); x = np.array(trim_values, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (pooled - pooled.min()) / (pooled.max() - pooled.min() + 1e-12)
    p0 = np.array([x_n[0], y_n[0]]); p1 = np.array([x_n[-1], y_n[-1]])
    line_vec = p1 - p0; line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12: return trim_values[int(np.nanargmin(pooled))]
    line_unit = line_vec / line_len
    dists = []
    for i in range(len(x_n)):
        pt = np.array([x_n[i], y_n[i]]) - p0
        perp = pt - np.dot(pt, line_unit) * line_unit
        dists.append(np.linalg.norm(perp))
    return trim_values[int(np.argmax(dists))]

def compute_stability_plateau(results, trim_values, tol_pct=5.0):
    pooled, opt = compute_pooled_median(results, trim_values)
    pooled = np.array(pooled); min_val = np.nanmin(pooled)
    threshold = min_val * (1 + tol_pct / 100.0)
    stable = [tv for tv, m in zip(trim_values, pooled) if m <= threshold]
    return (min(stable), max(stable)) if stable else (opt, opt)

def compute_normalised_composite(results, trim_values):
    zones = list(results.keys())
    zone_curves = {}
    for z in zones:
        meds = []
        for tv in trim_values:
            if tv in results[z]:
                v = results[z][tv]; v = v[~np.isnan(v)]
                meds.append(np.median(v) if len(v) > 0 else np.nan)
            else: meds.append(np.nan)
        zone_curves[z] = np.array(meds)
    norm_curves = {}
    for z in zones:
        c = zone_curves[z]; lo, hi = np.nanmin(c), np.nanmax(c)
        rng = hi - lo if hi - lo > 1e-12 else 1.0
        norm_curves[z] = (c - lo) / rng
    composite = np.nanmean(np.stack(list(norm_curves.values())), axis=0)
    return composite.tolist(), norm_curves, trim_values[int(np.nanargmin(composite))]

def compute_pareto_front(results, trim_values):
    zones = list(results.keys())
    zone_medians = {}
    for z in zones:
        meds = []
        for tv in trim_values:
            if tv in results[z]:
                v = results[z][tv][~np.isnan(results[z][tv])]
                meds.append(np.median(v) if len(v) > 0 else np.nan)
            else: meds.append(np.nan)
        zone_medians[z] = np.array(meds)
    n = len(trim_values); pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j: continue
            if (all(zone_medians[z][j] <= zone_medians[z][i] for z in zones) and
                any(zone_medians[z][j] <  zone_medians[z][i] for z in zones)):
                dominated = True; break
        if not dominated: pareto.append(trim_values[i])
    ideal = {z: np.nanmin(zone_medians[z]) for z in zones}
    best_tv, best_dist = pareto[0], np.inf
    for tv in pareto:
        idx = trim_values.index(tv)
        d = np.sqrt(sum((zone_medians[z][idx] - ideal[z])**2 for z in zones))
        if d < best_dist: best_dist = d; best_tv = tv
    return pareto, best_tv

def find_bic_optimal_trim(results, trim_values,
                          k_max=5, delta_bic_threshold=6.0,
                          plateau_fraction=0.10):
    pooled, _ = compute_pooled_median(results, trim_values)
    pooled = np.array(pooled); tv_arr = np.array(trim_values, dtype=float); n = len(tv_arr)
    def _fit_pw(bp_indices):
        bp = sorted(bp_indices)
        bounds = [0] + [b + 1 for b in bp]; ends = [b + 1 for b in bp] + [n]
        sse, n_params = 0.0, 0
        for s, e in zip(bounds, ends):
            if e - s < 2: return np.inf, np.inf
            xs, ys = tv_arr[s:e], pooled[s:e]; p = np.polyfit(xs, ys, 1)
            sse += np.sum((ys - np.polyval(p, xs))**2); n_params += 2
        return sse, n_params
    def _bic(sse, n_obs, n_p):
        if sse <= 0 or n_obs <= n_p: return np.inf
        return n_obs * np.log(sse / n_obs) + n_p * np.log(n_obs)
    bic_per_k, bps_per_k = [], []
    for k in range(0, k_max + 1):
        if k == 0:
            sse, np_ = _fit_pw([]); bic_per_k.append(_bic(sse, n, np_))
            bps_per_k.append([]); continue
        candidates = list(range(2, n - 2)); best_bic, best_bps = np.inf, []
        for combo in combinations(candidates, k):
            bp_s = sorted(combo); valid, prev = True, 0
            for b in bp_s:
                if b - prev < 2: valid = False; break
                prev = b + 1
            if n - prev < 2: valid = False
            if not valid: continue
            sse, np_ = _fit_pw(list(combo)); bic = _bic(sse, n, np_)
            if bic < best_bic: best_bic, best_bps = bic, list(combo)
        bic_per_k.append(best_bic); bps_per_k.append(best_bps)
    bic_table, selected_k = [], 0
    for k in range(len(bic_per_k)):
        delta = bic_per_k[k-1] - bic_per_k[k] if k > 0 else 0.0
        bic_table.append((k, bic_per_k[k], delta))
        if k > 0 and delta > delta_bic_threshold: selected_k = k
    sel_bps = sorted(bps_per_k[selected_k])
    bp_trims = [int(tv_arr[i]) for i in sel_bps]
    bounds_seg = [0] + [b + 1 for b in sel_bps] + [n]
    segments, slopes = [], []
    for seg in range(len(bounds_seg) - 1):
        s, e = bounds_seg[seg], bounds_seg[seg + 1]
        xs, ys = tv_arr[s:e], pooled[s:e]; p = np.polyfit(xs, ys, 1)
        slopes.append(p[0]); segments.append((int(tv_arr[s]), int(tv_arr[e-1]), p[0]))
    steepest = min(slopes) if slopes else 0
    plateau_thresh = plateau_fraction * abs(steepest)
    opt_trim = bp_trims[-1] if bp_trims else int(tv_arr[int(np.argmin(pooled))])
    for i, bp_idx in enumerate(sel_bps):
        if i + 1 < len(slopes) and abs(slopes[i + 1]) < plateau_thresh:
            opt_trim = int(tv_arr[bp_idx]); break
    total_imp = pooled[0] - np.nanmin(pooled)
    opt_list_idx = list(tv_arr).index(opt_trim)
    imp = pooled[0] - pooled[opt_list_idx]
    imp_pct = (imp / total_imp * 100) if total_imp > 0 else 0.0
    return {'opt_trim': opt_trim, 'n_breakpoints': selected_k,
            'breakpoints': bp_trims, 'segments': segments, 'bic_table': bic_table,
            'improvement_pct': imp_pct, 'w1norm_at_opt': pooled[opt_list_idx],
            'plateau_threshold': plateau_thresh, 'steepest_slope': abs(steepest)}


# %% Section 10: Calibration report, plotting, and user prompt
# =========================================================================

def _zone_label(z):
    """Clean zone key → display label (e.g. 'offshore_grab_samples' → 'Offshore')."""
    for key in ['offshore', 'channel', 'flats', 'all']:
        if key in z.lower():
            return key.capitalize()
    return z.replace('_', ' ').title()


def report_optimal_trim(results, trim_values):
    zones = list(results.keys())
    pooled_medians, opt_pooled = compute_pooled_median(results, trim_values)
    opt_knee = compute_kneedle(results, trim_values)
    zone_optima = compute_per_zone_minima(results, trim_values)
    stab_lo, stab_hi = compute_stability_plateau(results, trim_values)
    composite, norm_curves, opt_norm = compute_normalised_composite(results, trim_values)
    pareto_set, opt_pareto = compute_pareto_front(results, trim_values)
    bic_result = find_bic_optimal_trim(results, trim_values)
    opt_bic = bic_result['opt_trim']

    print("\n" + "=" * 72)
    print("  TRIMMING CALIBRATION REPORT")
    print("=" * 72)
    print("\n  Per-zone optima (criterion 3 — median W1norm):")
    for z in zones:
        tv, val = zone_optima[z]
        print(f"    {z:12s}:  trim = {tv:>2d}%   (median W1norm = {val:.4f})")
    print(f"\n  {'':12s}   Combined pooled minimum = {opt_pooled}%")
    print("\n  " + "-" * 68)
    print("  Criterion 2 — Knee / elbow detection (Kneedle)")
    print(f"    Optimal trim (knee point) = {opt_knee}%")
    print("\n  " + "-" * 68)
    print("  Criterion 4 — Stability plateau (pooled median within 5% of min)")
    print(f"    Stable range: {stab_lo}% – {stab_hi}%")
    print("\n  " + "-" * 68)
    print("  Criterion 5 — Normalised multi-objective composite")
    print(f"    Optimal trim = {opt_norm}%   (composite score = {composite[trim_values.index(opt_norm)]:.4f})")
    print("\n  " + "-" * 68)
    print("  Criterion 6 — Pareto-front analysis")
    print(f"    Pareto-optimal values: {', '.join(f'{p}%' for p in pareto_set)}")
    print(f"    Selected (closest to ideal) = {opt_pareto}%")
    print("\n  " + "-" * 68)
    print("  Criterion 8 — BIC-penalised segmented regression")
    print(f"    Model selection (ΔBIC > 6 = justified, Kass & Raftery 1995):")
    print(f"    {'k':>5s}  {'BIC':>10s}  {'ΔBIC':>10s}  Status")
    for k, bval, delta in bic_result['bic_table']:
        status = ("" if k == 0 else "justified" if delta > 6
                  else "marginal" if delta > 2 else "not justified")
        print(f"    {k:>5d}  {bval:>10.2f}  {delta:>10.2f}  {status}")
    print(f"\n    Selected: {bic_result['n_breakpoints']} breakpoint(s) at {bic_result['breakpoints']}%")
    for start, end, slope in bic_result['segments']:
        print(f"      {start:>2d}–{end:>2d}%:  slope = {slope:.5f} φ/%")
    print(f"    >>> BIC recommended trim (plateau onset) = {opt_bic}%")
    print(f"        W1norm at optimum = {bic_result['w1norm_at_opt']:.4f}")
    print(f"        Improvement captured: {bic_result['improvement_pct']:.1f}%")
    unique = set([opt_pooled, opt_knee, opt_norm, opt_pareto, opt_bic])
    print("\n  " + "=" * 68)
    if len(unique) == 1:
        print(f"  >>> CONSENSUS: all criteria agree on trim = {opt_pooled}% <<<")
    else:
        print(f"  >>> Criteria suggest trim values in the range {min(unique)}% – {max(unique)}% <<<")
        print(f"      Pooled minimum (global min) : {opt_pooled}%")
        print(f"      Knee / elbow               : {opt_knee}%")
        print(f"      Stability plateau           : {stab_lo}% – {stab_hi}%")
        print(f"      Normalised composite        : {opt_norm}%")
        print(f"      Pareto ideal-point          : {opt_pareto}%")
        print(f"      BIC segmented regression    : {opt_bic}%")
    print("=" * 72 + "\n")
    return {'pooled_minimum': opt_pooled, 'kneedle': opt_knee,
            'zone_optima': zone_optima, 'stability_range': (stab_lo, stab_hi),
            'normalised_composite': opt_norm, 'pareto_ideal': opt_pareto,
            'bic_optimal': opt_bic, 'bic_result': bic_result,
            'pareto_set': pareto_set, 'composite_scores': composite,
            'pooled_medians': pooled_medians}

def plot_calibration(results, trim_values, optima, output_folder, zone_optima=None):
    if not HAS_MPL:
        print("  matplotlib not available — skipping plot."); return
    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})
    zones = list(results.keys())
    # match zone names by substring so 'offshore_grab_samples' → 'offshore'
    _bg_base = {'offshore': '#cce5ff', 'channel': '#d4edda', 'flats': '#f8d7da', 'all': '#f0f0f0'}
    def _zone_bg(z):
        for key, col in _bg_base.items():
            if key in z.lower():
                return col
        return '#f0f0f0'
    n_zones = len(zones)
    fig, axes = plt.subplots(n_zones, 1, figsize=(8, 3.5 * n_zones), sharex=True, squeeze=False)
    for ax_idx, zone in enumerate(zones):
        ax = axes[ax_idx, 0]; bg = _zone_bg(zone); ax.set_facecolor(bg)
        bp_data, bp_pos = [], []
        for tv in trim_values:
            if tv in results[zone]:
                v = results[zone][tv]; v = v[~np.isnan(v)]
                if len(v) > 0: bp_data.append(v); bp_pos.append(tv)
        if bp_data:
            bplot = ax.boxplot(bp_data, positions=bp_pos, widths=0.6,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color='black', linewidth=1.2))
            for patch in bplot['boxes']: patch.set_facecolor(bg); patch.set_edgecolor('grey')
        medians = []
        for tv in trim_values:
            if tv in results[zone]:
                v = results[zone][tv][~np.isnan(results[zone][tv])]
                medians.append(np.median(v) if len(v) > 0 else np.nan)
            else: medians.append(np.nan)
        ax.plot(trim_values, medians, 'k-o', markersize=3, linewidth=1, label='Zone median')
        ax.set_ylabel(f'{_zone_label(zone)}\nW1norm (–)')
        ax.set_xlim(trim_values[0] - 1, trim_values[-1] + 1)
    ax = axes[-1, 0]
    op, ok, on, opar, ob = (optima['pooled_minimum'], optima['kneedle'],
        optima['normalised_composite'], optima['pareto_ideal'], optima['bic_optimal'])
    slo, shi = optima['stability_range']
    ax.axvline(op, color='red', ls='--', lw=1.2, alpha=0.8, label=f'Pooled min ({op}%)')
    ax.axvline(ok, color='blue', ls=':', lw=1.2, alpha=0.8, label=f'Kneedle ({ok}%)')
    if on not in (op, ok):
        ax.axvline(on, color='green', ls='-.', lw=1.2, alpha=0.8, label=f'Composite ({on}%)')
    if opar not in (op, ok, on):
        ax.axvline(opar, color='purple', ls='--', lw=1.2, alpha=0.8, label=f'Pareto ({opar}%)')
    if ob not in (op, ok, on, opar):
        ax.axvline(ob, color='orange', ls='-.', lw=1.2, alpha=0.8, label=f'BIC plateau ({ob}%)')
    ax.axvspan(slo, shi, alpha=0.08, color='grey', label=f'Stability ({slo}–{shi}%)')
    ax.set_xlabel('Trimming proportion α (%)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    if output_folder:
        ensure_dir(output_folder)
        fpath = os.path.join(output_folder, 'trimming_calibration.jpg')
        plt.savefig(fpath, dpi=600, bbox_inches='tight')
        print(f"  Calibration figure saved to {fpath}")
    plt.show()


def plot_calibration_summary(results, trim_values, optima, output_folder):
    """Single-panel summary with overlaid median lines per zone.

    Complements the per-zone boxplot by showing all zones and the
    pooled median on one set of axes for cross-zone comparison.
    """
    if not HAS_MPL:
        return

    zones = list(results.keys())
    if len(zones) <= 1:
        return  # summary only useful with multiple zones

    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10})

    zone_color_base = {
        'offshore': '#1f77b4', 'channel': '#2ca02c', 'flats': '#d62728',
        'all': 'black',
    }
    zone_style_base = {
        'offshore': '-', 'channel': '-', 'flats': '-', 'all': '--',
    }
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']

    def _zone_color(z, idx):
        for key, col in zone_color_base.items():
            if key in z.lower():
                return col
        return fallback_colors[idx % len(fallback_colors)]

    def _zone_style(z):
        for key, ls in zone_style_base.items():
            if key in z.lower():
                return ls
        return '-'

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, zone in enumerate(zones):
        medians = []
        for tv in trim_values:
            if tv in results[zone]:
                v = results[zone][tv]; v = v[~np.isnan(v)]
                medians.append(np.median(v) if len(v) > 0 else np.nan)
            else:
                medians.append(np.nan)

        color = _zone_color(zone, i)
        ls = _zone_style(zone)
        lw = 2.0 if 'all' in zone.lower() else 1.2
        marker = 's' if 'all' in zone.lower() else 'o'
        ax.plot(trim_values, medians, color=color, ls=ls, lw=lw,
                marker=marker, markersize=4, label=_zone_label(zone))

    # criteria markers
    op = optima['pooled_minimum']
    ok = optima['kneedle']
    ob = optima['bic_optimal']
    slo, shi = optima['stability_range']

    ax.axvline(op, color='red', ls='--', lw=1.0, alpha=0.7,
               label=f'Pooled min ({op}%)')
    ax.axvline(ok, color='blue', ls=':', lw=1.0, alpha=0.7,
               label=f'Kneedle ({ok}%)')
    if ob not in (op, ok):
        ax.axvline(ob, color='orange', ls='-.', lw=1.0, alpha=0.7,
                   label=f'BIC ({ob}%)')
    ax.axvspan(slo, shi, alpha=0.06, color='grey',
               label=f'Stability ({slo}–{shi}%)')

    ax.set_xlabel('Trimming proportion α (%)')
    ax.set_ylabel('Median W1norm (–)')
    ax.set_xlim(trim_values[0] - 0.5, trim_values[-1] + 0.5)
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    plt.tight_layout()

    if output_folder:
        ensure_dir(output_folder)
        fpath = os.path.join(output_folder, 'trimming_calibration_summary.jpg')
        plt.savefig(fpath, dpi=600, bbox_inches='tight')
        print(f"  Summary figure saved to {fpath}")
    plt.show()


def save_calibration_report_txt(optima, trim_values, output_folder):
    if output_folder is None: return
    ensure_dir(output_folder)
    fpath = os.path.join(output_folder, 'trimming_calibration_report.txt')
    with open(fpath, 'w') as f:
        f.write("TRIMMING CALIBRATION REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Sweep range: {trim_values[0]}–{trim_values[-1]}% "
                f"(step {trim_values[1] - trim_values[0]}%)\n\n")
        f.write(f"Pooled minimum (global min) : {optima['pooled_minimum']}%\n")
        f.write(f"Knee / elbow (Kneedle)      : {optima['kneedle']}%\n")
        stab = optima['stability_range']
        f.write(f"Stability plateau (±5%)     : {stab[0]}% – {stab[1]}%\n")
        f.write(f"Normalised composite        : {optima['normalised_composite']}%\n")
        f.write(f"Pareto ideal-point          : {optima['pareto_ideal']}%\n")
        f.write(f"BIC segmented regression    : {optima['bic_optimal']}%\n\n")
        for z, (tv, val) in optima['zone_optima'].items():
            f.write(f"Zone {z:12s}: optimal trim = {tv}%  (median W1norm = {val:.4f})\n")
        f.write(f"\nRecommended default (pooled minimum): {optima['pooled_minimum']}%\n")
    print(f"  Report saved to {fpath}")

def prompt_trim_alpha(optima):
    default = optima['pooled_minimum']
    print("=" * 72)
    print("  Enter your chosen trimming proportion (integer, %).")
    print(f"  Press Enter without typing to accept the pooled-minimum")
    print(f"  recommended value (α = {default}%).")
    print("=" * 72)
    try: raw = input(f"  >> Trimming proportion [{default}]: ").strip()
    except EOFError: raw = ""
    if raw == "":
        print(f"\n  Using recommended value: α = {default}%\n"); return default
    try:
        chosen = int(raw); print(f"\n  User selected: α = {chosen}%\n"); return chosen
    except ValueError:
        print(f"\n  Invalid input — defaulting to α = {default}%\n"); return default


# %% Section 11: Cache infrastructure
# =========================================================================

def _build_param_dict_calib(model_type, grain_sizes, shear_weight_method,
                            shear_weight_factor, w1norm_mode,
                            finest_phi, coarsest_phi, phi_interval,
                            input_mat_folder=None, input_grid_mat=None,
                            calib_trim_units=None, calib_trim_one_unit=None):
    """Parameters that affect calibration caches."""
    d = {'model_type': model_type, 'grain_sizes': sorted(grain_sizes),
         'shear_weight_method': shear_weight_method,
         'shear_weight_factor': float(shear_weight_factor),
         'w1norm_mode': w1norm_mode, 'finest_phi': float(finest_phi),
         'coarsest_phi': float(coarsest_phi),
         'phi_interval': float(phi_interval)}
    if input_mat_folder is not None:
        d['input_mat_folder'] = os.path.normpath(input_mat_folder)
    if input_grid_mat is not None:
        d['input_grid_mat'] = os.path.normpath(input_grid_mat)
    if calib_trim_units is not None:
        d['calib_trim_units'] = sorted(calib_trim_units)
    if calib_trim_one_unit is not None:
        d['calib_trim_one_unit'] = calib_trim_one_unit
    return d

def _build_param_dict_asf(model_type, trim_alpha, grain_sizes,
                           input_mat_folder=None):
    """Parameters that affect per-grain ASF caches.
    Shear weights excluded — they only affect the assembly step."""
    d = {'model_type': model_type, 'trim_alpha': int(trim_alpha),
         'grain_sizes': sorted(grain_sizes)}
    if input_mat_folder is not None:
        d['input_mat_folder'] = os.path.normpath(input_mat_folder)
    return d

def _param_hash(param_dict, length=8):
    """Deterministic short hash of a parameter dictionary."""
    canonical = json.dumps(param_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]

def _get_cache_dir(base_path, prefix, param_dict):
    """Return path: base_path/{prefix}_{hash}/"""
    return os.path.join(base_path, f"{prefix}_{_param_hash(param_dict)}")

def _verify_or_create_cache(cache_dir, param_dict):
    """Check manifest or create fresh cache.  Returns True if pre-existing and valid."""
    manifest = os.path.join(cache_dir, 'run_params.json')
    if os.path.isdir(cache_dir) and os.path.isfile(manifest):
        with open(manifest, 'r') as f:
            stored = json.load(f)
        if stored == param_dict:
            return True
        raise RuntimeError(
            f"Cache directory {cache_dir} exists but parameters differ.\n"
            f"  Stored:  {stored}\n  Current: {param_dict}\n"
            f"Delete the directory manually or use a different interim path.")
    os.makedirs(cache_dir, exist_ok=True)
    with open(manifest, 'w') as f:
        json.dump(param_dict, f, indent=2, sort_keys=True)
    return False


def _collect_all_params():
    """Collect every user-facing parameter into a single dict for logging.

    This captures the complete state of Section 2 so that any run can
    be reproduced exactly from its log file.
    """
    def _path(p):
        return os.path.normpath(p) if p else None

    return {
        '__ generated': datetime.now().isoformat(),
        '__ script': 'ASF Sediment Distribution Generator',
        # Section 2.1 — tasks
        'calib_trim': calib_trim,
        'asf_only': asf_only,
        'generate_d50': generate_d50,
        # Section 2.2 — calibration
        'calib_trim_one_unit': calib_trim_one_unit,
        'input_grab_sample_location': _path(input_grab_sample_location),
        'calib_trim_units': calib_trim_units,
        'trim_start': trim_start,
        'trim_end': trim_end,
        'trim_step': trim_step,
        'w1norm_mode': w1norm_mode,
        # Section 2.3 — directories
        'input_mat_folder': _path(input_mat_folder),
        'input_grid_mat': _path(input_grid_mat),
        'output_ASF_folder': _path(output_ASF_folder),
        'output_ASF_interim_folder': _path(output_ASF_interim_folder),
        'output_trim_calib_folder': _path(output_trim_calib_folder),
        'output_trim_interim_folder': _path(output_trim_interim_folder),
        'input_bss_mat': _path(input_bss_mat),
        # Section 2.4 — ASF parameters
        'max_workers': max_workers,
        'batch_size': batch_size,
        'trim_alpha': trim_alpha,
        'grain_sizes': grain_sizes,
        'finest_phi': finest_phi,
        'coarsest_phi': coarsest_phi,
        'phi_interval': phi_interval,
        'shear_weight_method': shear_weight_method,
        'shear_weight_factor': shear_weight_factor,
        # Section 2.5 — model
        'model_type': model_type,
        # Section 2.6 — experimental
        'operation_on_fluxes': operation_on_fluxes,
        'exceedance_pctile': exceedance_pctile,
        'analysis_on_operated_fluxes': analysis_on_operated_fluxes,
        'grain_size_thresh': grain_size_thresh,
    }


def _save_full_run_log(cache_dir, extra=None):
    """Write a comprehensive parameter log alongside the cache manifest.

    The file full_run_log.json records every user parameter for
    reproducibility.  'extra' can add run-specific info such as the
    final chosen trim_alpha after calibration.
    """
    log = _collect_all_params()
    if extra:
        log.update(extra)
    fpath = os.path.join(cache_dir, 'full_run_log.json')
    with open(fpath, 'w') as f:
        json.dump(log, f, indent=2, sort_keys=True, default=str)
    return fpath

def _calib_dist_path(cache_dir, trim_pct, model_t):
    d = os.path.join(cache_dir, 'distributions'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"asf_dist_trim{int(trim_pct)}_{model_t}.csv")

def _calib_w1norm_path(cache_dir, zone, trim_val):
    d = os.path.join(cache_dir, 'w1norm'); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"w1norm_{zone}_trim{trim_val}.npy")

def _asf_grain_path(cache_dir, mat_file):
    d = os.path.join(cache_dir, 'grain_caches'); os.makedirs(d, exist_ok=True)
    base = os.path.splitext(os.path.basename(mat_file))[0]
    return os.path.join(d, f"{base}.npz")

def _load_npz(path):
    if not os.path.isfile(path): return None
    try:
        d = np.load(path, allow_pickle=True)
        return (d['val_pos'], d['val_neg'], d['N_pos'], d['N_neg'], str(d['grain_size']))
    except Exception: return None

def _save_npz(path, vp, vn, np_, nn_, gs):
    np.savez_compressed(path, val_pos=vp, val_neg=vn,
                        N_pos=np_, N_neg=nn_, grain_size=gs)


# %% Section 12: Main execution
# =========================================================================

if __name__ == "__main__":

    st = time.time()
    print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    PHI_INTERVALS = generate_phi_intervals(finest_phi, coarsest_phi, phi_interval)
    print(f"Phi intervals: {PHI_INTERVALS}")
    print(f"Grain sizes:   {grain_sizes} µm")

    # ================================================================
    # TRIMMING CALIBRATION
    # ================================================================
    if calib_trim:
        calib_t0 = time.time()

        print("\n" + "=" * 72)
        print("  TRIMMING PROPORTION CALIBRATION")
        print(f"  W1norm mode: {w1norm_mode}")
        print("=" * 72)

        assert input_grab_sample_location is not None, "input_grab_sample_location must be set"
        assert calib_trim_units is not None, "calib_trim_units must contain at least one filename"
        assert input_mat_folder is not None, "input_mat_folder must be set"

        # --- parameter-verified cache ---
        calib_params = _build_param_dict_calib(
            model_type, grain_sizes, shear_weight_method,
            shear_weight_factor, w1norm_mode,
            finest_phi, coarsest_phi, phi_interval,
            input_mat_folder=input_mat_folder,
            input_grid_mat=input_grid_mat,
            calib_trim_units=calib_trim_units,
            calib_trim_one_unit=calib_trim_one_unit)
        calib_cache_dir = None
        if output_trim_interim_folder:
            calib_cache_dir = _get_cache_dir(output_trim_interim_folder, 'calib', calib_params)
            existed = _verify_or_create_cache(calib_cache_dir, calib_params)
            if existed:
                # count how many w1norm files actually exist
                w1_dir = os.path.join(calib_cache_dir, 'w1norm')
                n_cached = len(glob.glob(os.path.join(w1_dir, '*.npy'))) if os.path.isdir(w1_dir) else 0
                print(f"  Cache found: {calib_cache_dir}")
                print(f"  Parameters match — {n_cached} cached W1norm file(s) available.")
            else:
                print(f"  New cache: {calib_cache_dir}")
            _save_full_run_log(calib_cache_dir)

        print("\n  Loading validation data...")
        validation_data = load_validation_data(
            input_grab_sample_location, calib_trim_units,
            grain_sizes, one_unit=calib_trim_one_unit, phi_ivls=PHI_INTERVALS)
        for z, df in validation_data.items():
            print(f"    {z}: {len(df)} points")

        trim_values = list(range(trim_start, trim_end + 1, trim_step))
        print(f"\n  Sweep: {trim_values[0]}% → {trim_values[-1]}%, "
              f"step {trim_step}%, {len(trim_values)} iterations")

        results_calib = {z: {} for z in validation_data}

        for tv in tqdm(trim_values, desc="  Calibration sweep", ncols=80):
            # check W1norm cache
            all_w1_cached = True
            for zone in validation_data:
                if calib_cache_dir:
                    cp = _calib_w1norm_path(calib_cache_dir, zone, tv)
                    if os.path.isfile(cp):
                        results_calib[zone][tv] = np.load(cp)
                    else:
                        all_w1_cached = False
                else:
                    all_w1_cached = False
            if all_w1_cached:
                continue

            # check / generate distribution
            model_df = None
            if calib_cache_dir:
                dp = _calib_dist_path(calib_cache_dir, tv, model_type)
                if os.path.isfile(dp):
                    model_df = pd.read_csv(dp)
            if model_df is None:
                model_df, _, _ = generate_asf_distribution_fresh(
                    input_mat_folder, float(tv), model_type, grain_sizes,
                    shear_weight_method, shear_weight_factor,
                    grid_mat_path=input_grid_mat)
                if calib_cache_dir:
                    dp = _calib_dist_path(calib_cache_dir, tv, model_type)
                    model_df.to_csv(dp, index=False)

            # compute W1norm per zone
            for zone, val_df in validation_data.items():
                if zone in results_calib and tv in results_calib[zone]:
                    continue
                w1arr = compute_zonal_w1norm(model_df, val_df, grain_sizes,
                                            mode=w1norm_mode, phi_ivls=PHI_INTERVALS)
                results_calib[zone][tv] = w1arr
                if calib_cache_dir:
                    np.save(_calib_w1norm_path(calib_cache_dir, zone, tv), w1arr)

        optima = report_optimal_trim(results_calib, trim_values)
        save_calibration_report_txt(optima, trim_values, output_trim_calib_folder)
        plot_calibration(results_calib, trim_values, optima,
                         output_trim_calib_folder,
                         zone_optima=optima.get('zone_optima'))
        plot_calibration_summary(results_calib, trim_values, optima,
                                 output_trim_calib_folder)

        calib_elapsed = (time.time() - calib_t0) / 60.0
        print(f"  Calibration completed in {calib_elapsed:.1f} minutes.\n")

        trim_alpha = prompt_trim_alpha(optima)

    # ================================================================
    # ASF MAIN PIPELINE
    # ================================================================
    asf_mtype = f"{model_type}_a{trim_alpha}"

    print("\n" + "=" * 72)
    print("  ASF MAIN PIPELINE")
    print(f"  Trimming proportion α = {trim_alpha}%")
    print(f"  Output tag: {asf_mtype}")
    print("=" * 72)

    assert input_mat_folder is not None, "input_mat_folder must be set"
    assert input_grid_mat is not None, "input_grid_mat must be set"
    os.chdir(input_mat_folder)

    um_files = find_files(input_mat_folder, f"*um_{model_type}.mat")
    hyd_grid = find_files(
        os.path.dirname(input_grid_mat) if os.path.isfile(input_grid_mat) else input_mat_folder,
        os.path.basename(input_grid_mat) if os.path.isfile(input_grid_mat) else 'hyd_grid_nf.mat')
    assert len(um_files) > 0, f"No *um_{model_type}.mat files found"
    assert len(hyd_grid) > 0, "Hydrodynamic grid file not found"

    x_grid, y_grid = load_grid_coordinates(hyd_grid[0])

    bss_all = None
    if not asf_only and operation_on_fluxes == 8:
        assert input_bss_mat is not None, "input_bss_mat required for operation 8"
        with h5py.File(input_bss_mat, "r") as fb:
            bss_all = fb["data"]["Val"][:].astype(np.float32)
            print(f"  BSS loaded: {bss_all.shape}")

    ensure_dir(output_ASF_folder)

    # --- parameter-verified ASF cache ---
    asf_params = _build_param_dict_asf(model_type, trim_alpha, grain_sizes,
                                       input_mat_folder=input_mat_folder)
    asf_cache_dir = None
    if output_ASF_interim_folder:
        asf_cache_dir = _get_cache_dir(output_ASF_interim_folder, 'asf', asf_params)
        existed = _verify_or_create_cache(asf_cache_dir, asf_params)
        if existed:
            g_dir = os.path.join(asf_cache_dir, 'grain_caches')
            n_cached = len(glob.glob(os.path.join(g_dir, '*.npz'))) if os.path.isdir(g_dir) else 0
            print(f"  Cache found: {asf_cache_dir}")
            print(f"  Parameters match — {n_cached} cached grain file(s) available.")
        else:
            print(f"  New cache: {asf_cache_dir}")
        _save_full_run_log(asf_cache_dir,
                           extra={'trim_alpha_final': trim_alpha})

    val_pos_dict, val_neg_dict = {}, {}
    N_pos_dict, N_neg_dict = {}, {}
    lock = Lock()

    def _process_asf_file(mat_path):
        if asf_cache_dir:
            cp = _asf_grain_path(asf_cache_dir, mat_path)
            cached = _load_npz(cp)
            if cached is not None: return cached
        vp, vn, np_, nn_, gs = compute_trimmed_mean_flux(mat_path, trim_alpha)
        if asf_cache_dir:
            _save_npz(_asf_grain_path(asf_cache_dir, mat_path), vp, vn, np_, nn_, gs)
        return vp, vn, np_, nn_, gs

    print(f"\n  Processing {len(um_files)} grain-size files "
          f"(workers={max_workers}, batch={batch_size})...")

    for i in range(0, len(um_files), batch_size):
        batch = um_files[i:i + batch_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_asf_file, f): f for f in batch}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc=f'  Batch {i // batch_size + 1}'):
                fname = futures[future]
                try:
                    vp, vn, np_, nn_, gs = future.result()
                    with lock:
                        val_pos_dict[fname] = vp; val_neg_dict[fname] = vn
                        N_pos_dict[fname] = np_; N_neg_dict[fname] = nn_
                except Exception as e:
                    print(f"  Error processing {fname}: {e}")
        gc.collect()

    sorted_files = sorted(um_files, key=extract_sort_key)
    missing = [f for f in sorted_files if f not in val_pos_dict]
    if missing:
        print(f"\n  WARNING: {len(missing)} file(s) not processed:")
        for m in missing: print(f"    - {os.path.basename(m)}")
        raise RuntimeError(f"{len(missing)} grain-size file(s) missing — cannot proceed.")

    val_pos_list = [val_pos_dict[f] for f in sorted_files]
    val_neg_list = [val_neg_dict[f] for f in sorted_files]
    N_pos_list   = [N_pos_dict[f]   for f in sorted_files]
    N_neg_list   = [N_neg_dict[f]   for f in sorted_files]

    print("\n  Assembling ASF (Equation 4)...")
    dataset = assemble_asf(val_pos_list, val_neg_list, N_pos_list, N_neg_list,
                           grain_sizes, shear_weight_method, shear_weight_factor)

    asf_titles = []
    for f in sorted_files:
        gs = re.search(r"(\d+)um", f)
        asf_titles.append(f"ASF_a{trim_alpha}_{gs.group(1) if gs else 'unknown'}um")

    # ================================================================
    # EXPERIMENTAL APPROACHES
    # ================================================================
    exp_dataset, exp_titles = None, []
    if not asf_only:
        print(f"\n  Running experimental flux analysis "
              f"(op={operation_on_fluxes}, analysis={analysis_on_operated_fluxes})...")
        exp_pos_dict, exp_neg_dict, exp_cfr_dict = {}, {}, {}
        exp_pos_titles, exp_neg_titles, exp_cfr_titles = {}, {}, {}
        for mat_path in tqdm(sorted_files, desc='  Experimental ops'):
            if analysis_on_operated_fluxes in {1, 2, 3, 4, 5}:
                vp, tp, _ = operate_mat(mat_path, operation_on_fluxes, 1,
                    analysis_on_operated_fluxes, exceed_pct=exceedance_pctile,
                    bss_array=bss_all if operation_on_fluxes == 8 else None)
                vn, tn, _ = operate_mat(mat_path, operation_on_fluxes, 2,
                    analysis_on_operated_fluxes, exceed_pct=exceedance_pctile,
                    bss_array=bss_all if operation_on_fluxes == 8 else None)
                exp_pos_dict[mat_path] = vp; exp_neg_dict[mat_path] = vn
                exp_pos_titles[mat_path] = tp; exp_neg_titles[mat_path] = tn
            if analysis_on_operated_fluxes == 6:
                cfr, tc, _ = operate_mat(mat_path, operation_on_fluxes, 3,
                    analysis_on_operated_fluxes, exceed_pct=exceedance_pctile,
                    bss_array=bss_all if operation_on_fluxes == 8 else None)
                exp_cfr_dict[mat_path] = cfr; exp_cfr_titles[mat_path] = tc
        exp_dataset, exp_titles = process_experimental_flux(
            analysis=analysis_on_operated_fluxes,
            val_pos_list=[exp_pos_dict[f] for f in sorted_files if f in exp_pos_dict],
            val_neg_list=[exp_neg_dict[f] for f in sorted_files if f in exp_neg_dict],
            cfr_list=[exp_cfr_dict[f] for f in sorted_files if f in exp_cfr_dict],
            pos_title=[exp_pos_titles[f] for f in sorted_files if f in exp_pos_titles],
            neg_title=[exp_neg_titles[f] for f in sorted_files if f in exp_neg_titles],
            cfr_title=[exp_cfr_titles[f] for f in sorted_files if f in exp_cfr_titles],
            grain_sizes_um=grain_sizes,
            weight_method=shear_weight_method, weight_factor=shear_weight_factor,
            thresh=grain_size_thresh)

    # ================================================================
    # WRITE OUTPUTS
    # ================================================================
    print("\n  Writing output files...")

    def _write_dataset(ds, titles, out_path, x_arr, y_arr, m_type):
        ref_shape = ds[:, :, 0].shape
        xc, yc = align_grid_to_data(x_arr, y_arr, ref_shape)

        # 1. raw flux — track written paths so we never re-glob the folder
        written_flux = []
        for k in tqdm(range(ds.shape[2]), desc='    Raw flux files', leave=True):
            val = ds[:, :, k]
            xyz = np.vstack((xc.flatten(), yc.flatten(), val.flatten())).T
            fpath = os.path.join(out_path, f"{safe_filename(titles[k])}.xyz")
            np.savetxt(fpath, xyz, fmt="%f %f %.12f", header="X Y Z")
            written_flux.append(fpath)
        print("    Raw flux files written.")

        # 2. normalise — use tracked paths, not a glob
        process_files = sorted(written_flux,
                               key=lambda fn: int(re.search(r"(\d+)um", fn).group(1)))
        if len(process_files) == 0:
            print("    No flux files written — skipping normalisation."); return

        col_names = [re.search(r"(\d+)um", n).group(1) for n in process_files]
        df = combine_tabular_files(process_files, sep=' ', skiprows=1)
        df.columns = col_names
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0)
        row_sum = df.sum(axis=1)
        df.loc[row_sum == 0, df.columns] = 1.0 / len(df.columns)
        row_sum = df.sum(axis=1)
        df = df.div(row_sum, axis=0) * 100.0
        diff = 100.0 - df.sum(axis=1)
        df = df.add(diff / len(df.columns), axis=0)

        xy = pd.read_csv(process_files[0], sep=' ', skiprows=1, header=None)
        xy.columns = ['x', 'y', 'z']; xy = xy.drop('z', axis=1)

        # write perc files and track their paths
        written_perc = []
        for col in tqdm(df.columns, desc='    Normalised % files', leave=True):
            out = xy.copy(); out['z'] = df[col].values
            fname = f"{col}_perc_{m_type}.xyz"
            fpath = os.path.join(out_path, fname)
            out.to_csv(fpath, sep=',', index=False, header=False, float_format="%.12f")
            written_perc.append(fpath)
        print("    Normalised percentage files written.")

        # 3. D50
        if generate_d50:
            print("\n    Generating D50 map...")
            gs_str = [str(int(g)) for g in grain_sizes]
            valid_cols = [c for c in gs_str if c in df.columns]
            if len(valid_cols) > 0:
                d50_df = sp_interp(df[valid_cols], grain_sizes, 50, PHI_INTERVALS)
                xy_d50 = xy.copy(); xy_d50['z'] = d50_df['D50']
                fname = f"D50_{m_type}_{'-'.join(valid_cols)}.xyz"
                xy_d50.to_csv(os.path.join(out_path, fname), sep=',', index=False, header=False)
                print(f"    D50 written to {fname}")

        # 4. bed-layer fractions — use tracked perc paths
        if len(written_perc) == 0:
            print("    No percentage files — skipping bed-layer step."); return

        perc_cols = []
        for name in written_perc:
            m = re.search(r"(\d+)_perc_" + re.escape(m_type) + r".xyz", name)
            if m: perc_cols.append(m.group(1))

        perc_df = combine_tabular_files(written_perc, sep=',', skiprows=0)
        perc_df.columns = perc_cols
        perc_df = perc_df / 100.0
        zero_rows = perc_df.sum(axis=1) == 0
        perc_df.loc[zero_rows, :] = 1.0 / len(perc_df.columns)
        perc_df = perc_df.div(perc_df.sum(axis=1), axis=0)
        perc_df.fillna(1.0 / len(perc_df.columns), inplace=True)

        xy_bl = pd.read_csv(written_perc[0], sep=',', header=None)
        xy_bl.columns = ['x', 'y', 'z']; xy_bl = xy_bl.drop('z', axis=1)

        for col in tqdm(perc_df.columns, desc='    Bed-layer fractions', leave=True):
            out_bl = xy_bl.copy(); out_bl['z'] = perc_df[col].values
            out_bl.to_csv(os.path.join(out_path, f"{col}_{m_type}_bl.xyz"),
                          sep=',', index=False, header=False)
        print("    Bed-layer fraction files written.")

    _write_dataset(dataset, asf_titles, output_ASF_folder, x_grid, y_grid, asf_mtype)

    if exp_dataset is not None:
        exp_out = os.path.join(output_ASF_folder, 'experimental')
        ensure_dir(exp_out)
        _write_dataset(exp_dataset, exp_titles, exp_out, x_grid, y_grid, model_type + '_exp')

    elapsed = time.time() - st

    # save complete run configuration to the output folder
    run_config = _collect_all_params()
    run_config['trim_alpha_final'] = trim_alpha
    run_config['asf_output_tag'] = asf_mtype
    run_config['elapsed_minutes'] = round(elapsed / 60, 2)
    run_config['phi_intervals'] = [list(iv) for iv in PHI_INTERVALS]
    config_path = os.path.join(output_ASF_folder, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2, sort_keys=True, default=str)
    print(f"\n  Run configuration saved to {config_path}")

    print(f"\n{'=' * 72}")
    print(f"  Finished in {elapsed / 60:.2f} minutes")
    print(f"{'=' * 72}")