# Accumulated Sediment Flux (ASF) — Sediment Distribution Generator

Post-processes **Delft3D-4** accumulated seabed flux output to generate a quasi-equilibrium sediment distribution based on the gross (erosion+deposition) bed material exchange. This is the companion code for:

> Soares et al. (under review), *"Process-based sediment mapping — an alternative for dynamic and data-scarce coasts"*, Applied Ocean Research.

---

## What it does

The script reads accumulated flux MAT files (one per grain-size class) and a hydrodynamic grid file exported from Delft3D-4 in HDF5/v7.3 format. It then:

1. Separates positive (depositional) and negative (erosional) flux time series per grain size per grid cell.
2. Applies symmetric trimming at a user-defined proportion α, retaining the central portion of the flux distribution before averaging (Equations 2–3 in the manuscript).
3. Combines the trimmed means using frequency-weighted gross transport; each flux direction contributes in proportion to the number of time steps that survived trimming (Equation 4).
4. Normalises the combined flux across grain sizes to produce percentage availability maps, bed-layer fractions (sum = 1), and an optional D50 map.
5. Writes all outputs as `.xyz` text files importable into GIS or plotting software.

An optional calibration workflow sweeps α over a user-defined range, evaluates each distribution against grab-sample validation data using the IQR-normalised Wasserstein distance (W₁,norm), and recommends an optimal α through several complementary criteria.

While the mixed layer concept is the default sediment model in Delft3D-4, the ASF requires the keyword ‘CumNetSedimentationFlux’ to be set as an output field in the .mor file to output the flux mat files. 


## Required inputs

| Input | Format | Description |
|-------|--------|-------------|
| Accumulated flux files | `{grain_size}um_{model_type}.mat` (HDF5) | One per grain-size class, exported from Delft3D-4 with `data/Val` containing the 3-D flux array (M × N × T) |
| Hydrodynamic grid | `*.mat` (HDF5) | Single time step, containing `data/X` and `data/Y` coordinate arrays. Handles both 2-D grids and 3-D grids with a k-layer dimension |

### Optional inputs

| Input | When needed |
|-------|-------------|
| Grab-sample CSV(s) | Trimming calibration. Columns: `X`, `Y`, `um_105`, `um_150`, … with distributional availability per fraction (proportions or percentages — auto-detected) |
| Bed-shear-stress MAT | Experimental operations 7–8 only |


## Outputs

| File pattern | Content |
|--------------|---------|
| `ASF_a{α}_{grain}um.xyz` | Raw ASF flux per grain size |
| `{grain}_perc_{tag}.xyz` | Normalised percentage availability |
| `{grain}_{tag}_bl.xyz` | Bed-layer fractions (sum = 1 per cell) |
| `D50_{tag}_{grains}.xyz` | Median grain-size map |
| `trimming_calibration.jpg` | Per-zone boxplot of W₁,norm vs α |
| `trimming_calibration_summary.jpg` | Overlaid median lines per zone |
| `trimming_calibration_report.txt` | Text summary of all selection criteria |
| `run_config.json` | Complete parameter log for reproducibility |


## Quick start

1. Export accumulated flux files from Delft3D-4 (one per grain-size class).
2. Export a single-timestep hydrodynamic grid file.
3. Open `main.py` and edit **Section 2** — all user parameters are in one block at the top:
   - Set paths to your MAT files and output directories.
   - Define your grain sizes and matching phi-scale bounds.
   - Choose whether to run calibration, and point to your validation CSV(s).
4. Run in Spyder (cell markers `# %%` throughout) or from the command line:
   ```
   python main.py
   ```
5. If calibration is enabled, the script sweeps the trimming range, prints a report with recommended α values, shows the calibration figures, and prompts you to choose. It then runs the full ASF pipeline with your chosen α.


## Trimming calibration

When `calib_trim = True`, the script:

- Computes ASF distributions for each α in the sweep range.
- Evaluates them against grab-sample data using W₁,norm (Wasserstein distance normalised by the observed IQR in φ-space).
- Reports optimal α from six complementary criteria:
  1. **Pooled minimum** — α minimising the pooled median W₁,norm across all zones.
  2. **Kneedle** — elbow detection on the pooled median curve.
  3. **Per-zone minima** — optimal α for each zone individually.
  4. **Stability plateau** — range of α where the pooled median stays within 5% of the minimum.
  5. **Normalised composite** — equal-weight average of min-max normalised per-zone medians.
  6. **BIC segmented regression** — piecewise-linear fit with Bayesian Information Criterion model selection (Kass & Raftery, 1995); plateau onset where slope falls below 10% of the steepest segment.

Validation data can be provided as a single file or as separate per-zone files. When multiple files are given, the script tracks each zone individually and adds a pooled result, producing both per-zone boxplots and a summary figure with overlaid median lines.


## Caching

Intermediate results are cached in parameter-verified subdirectories. Each cache directory contains:

- `run_params.json` — the parameters used to generate the cache (used for verification on subsequent runs).
- `full_run_log.json` — complete record of all user parameters.

The cache hash includes the input data path, grain sizes, phi settings, shear-weight configuration, W₁,norm mode, grid file, and validation file names. Changing any of these creates a new cache subdirectory; the old one is left untouched. On each run, the script reports how many cached files are available and only recomputes what is missing.


## Grid handling

The script handles several Delft3D grid configurations automatically:

- **2-D grids** where X/Y have shape (M, N) or (M+1, N+1).
- **3-D grids** with a leading k-layer dimension (K × M × N) — the first layer is used.
- **Asymmetric offsets** where the grid is +1 in only one dimension.

When the grid is larger than the data, cell-centre coordinates are computed by averaging adjacent corner nodes rather than arbitrarily trimming rows or columns.


## Grain-size subsetting

If the validation CSV contains more grain-size columns than specified in `grain_sizes`, the script:

- Extracts only the matching columns.
- Drops the rest with a printed warning.
- Renormalises the remaining fractions to sum = 1 per row.

This allows you to use the same grab-sample dataset across runs with different grain-size configurations.


## Exploratory combinations

Beyond the default ASF method, the script supports several experimental flux descriptors when `asf_only = False`. The ASF has its own dedicated pipeline and does not use these operation/analysis codes.

### Operations — time-series statistic per grain size and flux sign

| `op` | Description |
|------|-------------|
| 1 | Arithmetic mean |
| 2 | 95th percentile |
| 3 | Minimum |
| 4 | Maximum |
| 5 | Median |
| 6 | 75th percentile |
| 7 | Exceedance mean — mean of values above a user-set percentile |
| 8 | BSS-conditioned mean — mean at times when bed shear stress exceeds a percentile (requires BSS file) |

### Analyses — combination of positive and negative flux components

| `analysis` | Description |
|------------|-------------|
| 1 | Threshold split: positive for grain sizes ≤ threshold, negative for sizes above |
| 2 | Positive flux only (depositional) |
| 3 | Negative flux only (erosional, absolute value) |
| 4 | Residual depositional: positive − negative, clipped ≥ 0 |
| 5 | Geometric mean of positive and negative magnitudes |
| 6 | Cumulative flux ratio (CFR): ratio of positive to negative statistics |

### Recommended combinations

| Purpose | Setting | Notes |
|---------|---------|-------|
| **ASF sediment distribution** | `asf_only = True` | Default. Dedicated pipeline with trimmed mean + frequency weighting (Equations 2–4) |
| Depositional intensity | `op = 1/2/5, analysis = 2` | Positive flux only |
| Erosional intensity | `op = 1/2/5, analysis = 3` | Negative flux only (absolute) |
| Net residual deposition | `op = 1, analysis = 4` | Areas of persistent deposition |
| CFR diagnostics | `op = 1–8, analysis = 6` | Different ops give different CFR summaries |


## Dependencies
 
- Python ≥ 3.8
- `numpy`, `pandas`, `scipy`, `h5py`, `tqdm`, `psutil`
- `matplotlib`, `seaborn` (optional — for calibration figures; script runs headless without them)


## Citation

If you use this code, please cite:

> Soares et al. (under review), "Process-based sediment mapping — an alternative for dynamic and data-scarce coasts", *Applied Ocean Research*.
