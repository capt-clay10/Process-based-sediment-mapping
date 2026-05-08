# Accumulated Sediment Flux (ASF) — Sediment Distribution Generator

Post-processes **Delft3D-4** accumulated seabed-flux output to generate a relative seabed sediment distribution from frequency-weighted gross bed-material exchange. The workflow implements the Accumulated Sediment Flux (ASF) concept introduced in:

> Soares, C. C., Persichini, G., Galiforni-Silva, F., Herrling, G., & Winter, C. (2026). *Process-based sediment mapping – an alternative for dynamic and data-scarce coasts*. **Applied Ocean Research, 171**, 105085. https://doi.org/10.1016/j.apor.2026.105085

ASF is intended as a process-based mapping tool for dynamic, predominantly non-cohesive sandy coasts where representative hydrodynamic forcing and morphology are available but sediment samples are sparse, inaccessible, outdated, or absent.

---

## Contents

- [Scientific purpose](#scientific-purpose)
- [What the script does](#what-the-script-does)
- [When ASF is appropriate](#when-asf-is-appropriate)
- [Repository files](#repository-files)
- [Dependencies](#dependencies)
- [Required Delft3D-4 output](#required-delft3d-4-output)
- [How to use the script](#how-to-use-the-script)
- [Input data formats](#input-data-formats)
- [Main user parameters](#main-user-parameters)
- [Outputs](#outputs)
- [Trimming calibration](#trimming-calibration)
- [Validation grain-size harmonisation](#validation-grain-size-harmonisation)
- [Post-validation diagnostic plots](#post-validation-diagnostic-plots)
- [Caching and reproducibility](#caching-and-reproducibility)
- [Experimental flux descriptors](#experimental-flux-descriptors)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Scientific purpose

Sediment distribution maps are commonly produced by interpolating grab samples. In dynamic data-scarce coastal environments, however, sparse or temporally inconsistent samples may fail to capture morphology-driven and event-derived sediment redistribution. ASF addresses this by using Delft3D-4 accumulated sediment-flux time series to infer the relative availability of sediment fractions that are repeatedly entrained and redeposited under representative forcing conditions.

The method produces a **relative seabed sediment distribution**, not a morphodynamic forecast. It is best interpreted as a process-based map of dynamically active surface sediment fractions for the simulated forcing period. In simple words, the method shows you where a sediment could exist based on its mobility under governing conditions.

<img width="1519" height="590" alt="Image" src="https://github.com/user-attachments/assets/1706c8a8-15c2-4b6b-8d02-091145925854" />

Graphical abstract from Soares et al. (2026)

---

## What the script does

`main.py` reads accumulated flux MAT files exported from Delft3D-4 and a hydrodynamic grid MAT file. For each grain-size class and model grid cell, it:

1. Splits the accumulated flux time series into positive (depositional) and negative (erosional) components.
2. Applies a symmetric trimming proportion, α, to remove infrequent lower- and upper-tail flux values before averaging. A user calibration parameter. 
3. Computes positive and negative central trimmed means.
4. Weights each flux direction by the number of surviving positive and negative flux time steps.
5. Combines both signs into the ASF metric for each grain size.
6. Optionally applies critical-shear-stress weighting.
7. Normalises ASF values across grain sizes to generate relative percentage availability and bed-layer fraction files.
8. Optionally generates a D50 map.
9. Optionally calibrates α against grab-sample validation data using W1norm.
10. Optionally generates post-validation diagnostic plots.

The core ASF pipeline always uses the trimmed mean and frequency-weighted gross transport formulation. Experimental alternatives are available separately when `asf_only = False`.

---

## When ASF is appropriate

ASF is most suitable for:

- dynamic, predominantly non-cohesive sandy environments;
- coastlines where recent hydrodynamic forcing controls surface sediment cover;
- systems where sediment samples are sparse, outdated, spatially biased, or difficult to collect;
- applications requiring spatially continuous grain-size availability maps for habitat, ecotope, or coastal-management analyses.
- or if you want to know where sediments of a given range could naturally exist. 

ASF should be used cautiously in:

- low-energy zones dominated by inactive or relict sediments;
- cohesive or strongly mixed sand-mud environments unless the underlying Delft3D-4 sediment model adequately represents mud-specific processes;
- applications requiring stratigraphic preservation, hiding, exposure, or armouring processes;
- direct morphodynamic forecasting. ASF is a mapping/post-processing tool, not a replacement for a full morphodynamic bed-composition framework.

> In the ideal workflow, you would use ASF to generate an initial map of sediment distribution and then use a more process-preserving approach like the BCG (van der Wegen et al., 2011) on those results to run a morphodynamic simulation.

> van der Wegen, M., Dastgheib, A., Jaffe, B. E., & Roelvink, D. (2011). Bed composition generation for morphodynamic modeling: case study of San Pablo Bay in California, USA. Ocean dynamics, 61(2), 173-186. 

The published study found that ASF can reproduce morphology-conforming sorting patterns in dynamic sandy zones, but may show a fine-sediment bias when no prior sediment information is available. Relict coarse deposits in low-energy areas generally require geological priors or hybrid data-driven constraints.

---

## Repository files

| File | Purpose |
|------|---------|
| `main.py` | Main ASF post-processing, calibration, validation, plotting, and experimental-analysis script |
| `requirements.txt` | Core Python dependencies |
| `README.md` | User documentation |
| `LICENSE` | MIT License |

---

## Dependencies

### Core dependencies

Install the core dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The core dependencies are:

- `numpy`
- `pandas`
- `scipy`
- `h5py`
- `tqdm`
- `psutil`

### Optional plotting dependencies

Calibration and post-validation figures require:

```bash
pip install matplotlib seaborn
```

The script can still run without these packages, but plotting steps will be skipped.

### Suggested Python environment

Python 3.8 or newer is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install matplotlib seaborn
```

---

## Required Delft3D-4 output

ASF requires Delft3D-4 accumulated seabed sediment-flux output for each grain-size class. In the Delft3D-4 `.mor` file, the accumulated sediment-flux output must be enabled using:

```text
CumNetSedimentationFlux
```

The required accumulated flux files must be saved as HDF5/v7.3 MAT files with names following:

```text
{grain_size}um_{model_type}.mat
```

Examples:

```text
75um_news.mat
105um_news.mat
150um_news.mat
210um_news.mat
300um_news.mat
420um_news.mat
600um_news.mat
840um_news.mat
```

The `model_type` value in `main.py` must match the filename suffix. For the examples above:

```python
model_type = 'news'
```

Each flux MAT file must contain:

```text
data/Val
```

where `data/Val` is a 3-D array with dimensions equivalent to:

```text
M × N × T
```

where `M` and `N` are grid dimensions and `T` is the accumulated-flux output time dimension.

A hydrodynamic grid MAT file is also required. It must contain:

```text
data/X
data/Y
```

The script accepts 2-D grids and 3-D grids with a leading layer dimension. If the coordinate grid is one row or column larger than the flux data, the script computes cell-centre coordinates automatically.

---

## How to use the script

### 1. Prepare the Delft3D-4 simulation

Run a Delft3D-4 simulation for a representative hydrodynamic forcing period. The forcing period should correspond to the sediment-distribution state you want ASF to represent. For long-term  surface sediment mapping, use a climatologically representative forcing period (for example, Soares et al., 2024) rather than a single extreme event.

> Soares, C. C., Galiforni-Silva, F., & Winter, C. (2024). Representative residual transport pathways in a mixed-energy open tidal system. Journal of Sea Research, 201, 102530.

For the accumulated flux output, ensure that `CumNetSedimentationFlux` is included in the Delft3D-4 `.mor` output settings.

### 2. Export one flux MAT file per grain size

Place all accumulated flux files in one folder. File names must contain the grain size in micrometres and the model-type suffix:

```text
{grain_size}um_{model_type}.mat
```

For example, if `grain_sizes = [75, 105, 150, 210, 300, 420, 600, 840]` and `model_type = 'news'`, the folder should contain:

```text
75um_news.mat
105um_news.mat
150um_news.mat
210um_news.mat
300um_news.mat
420um_news.mat
600um_news.mat
840um_news.mat
```

The order of files does not need to be manual; the script sorts them by the grain-size value in the filename.

### 3. Prepare the hydrodynamic grid file

Provide a grid MAT file containing `data/X` and `data/Y`. The grid file may be in the same folder as the flux files or elsewhere, but the full path must be provided in Section 2 of `main.py`.

Example:

```python
input_grid_mat = r'D:/project/asf/news_hyd_grd.mat'
```

### 4. Edit Section 2 of `main.py`

All user settings are grouped near the top of `main.py` under **Section 2: User parameters**. For normal use, edit only this section.

#### Minimal ASF-only run

Use this configuration to run the ASF method without calibration:

```python
calib_trim = False
asf_only = True
generate_d50 = True
make_post_validation_plots = False

input_mat_folder = r'D:/project/asf/flux_mat_files'
input_grid_mat = r'D:/project/asf/news_hyd_grd.mat'
output_ASF_folder = r'D:/project/asf/output'
output_ASF_interim_folder = r'D:/project/asf/cache'

model_type = 'news'
grain_sizes = [75, 105, 150, 210, 300, 420, 600, 840]

finest_phi = 4.0
coarsest_phi = 0.0
phi_interval = 0.5

trim_alpha = 24
shear_weight_method = 0
shear_weight_factor = 1.0
```

Notes:

- `trim_alpha` is given as an integer percentage.
- The script default is `trim_alpha = 24`.
- The published ASF application used α = 0.25 for the data-scarce assessment, which corresponds to `trim_alpha = 25`.
- `grain_sizes` must match the grain sizes in the MAT filenames.
- The number of phi intervals generated by `finest_phi`, `coarsest_phi`, and `phi_interval` should match the number of grain-size classes.

### 5. Run the script

From the command line:

```bash
python main.py
```

You can also run the script in Spyder, VS Code, or another Python IDE. The script includes `# %%` cell markers for section-based execution in editors that support cells.

### 6. Run trimming calibration, if validation data are available

To calibrate α before the final ASF run, set:

```python
calib_trim = True
make_post_validation_plots = True

input_grab_sample_location = r'D:/project/asf/validation'
calib_trim_units = [
    'grabs_63to2000um.csv'
]

trim_start = 0
trim_end = 35
trim_step = 1
w1norm_mode = 'full'
```

Then run:

```bash
python main.py
```

The script will:

1. Load and harmonise the validation data.
2. Sweep α from `trim_start` to `trim_end` using `trim_step`.
3. Compute ASF distributions for each α.
4. Compare each distribution against validation data using W1norm.
5. Report several recommended α values.
6. Save calibration plots and a calibration report.
7. Prompt you to accept or manually enter the final α.
8. Run the final ASF pipeline using the selected α.

### 7. Use post-validation diagnostics

To generate post-validation plots after the final ASF run, provide grab-sample CSV files and set:

```python
make_post_validation_plots = True
```

This produces percentile-wise observed-vs-modelled plots, W1norm heatmaps, and aggregated W1norm boxplots.

### 8. Import outputs into GIS or modelling tools

The main spatial outputs are plain-text `.xyz` files. Percentage files are written as comma-separated `x,y,z` rows, and bed-layer files are written as fraction values that sum to 1 per grid cell.

Typical usage:

- import `D50_*.xyz` into GIS for median grain-size mapping;
- import `{grain}_perc_*.xyz` to analyse relative percentage availability by grain-size class;
- use `{grain}_{tag}_bl.xyz` as bed-layer fraction inputs where a fractional sediment composition is required.

---

## Input data formats

### Flux MAT files

| Requirement | Description |
|------------|-------------|
| File pattern | `{grain_size}um_{model_type}.mat` |
| Format | MATLAB v7.3 / HDF5 MAT file |
| Required dataset | `data/Val` |
| Expected dimensions | `M × N × T` |
| Sign convention | Positive values are treated as depositional flux; negative values are treated as erosional flux and converted to absolute magnitude |

### Hydrodynamic grid MAT file

| Requirement | Description |
|------------|-------------|
| Format | MATLAB v7.3 / HDF5 MAT file |
| Required datasets | `data/X`, `data/Y` |
| Accepted shapes | 2-D grid, 3-D grid with leading layer dimension, or grids offset by +1 in one or both horizontal dimensions |

### Validation CSV files

Validation files are optional and are used for trimming calibration and post-validation diagnostics.

Coordinate columns must be either:

```text
X,Y
```

or:

```text
x,y
```

Grain-size columns should follow:

```text
um_{grain_size}
```

Examples:

```text
X,Y,um_75,um_105,um_150,um_210,um_300,um_420,um_600,um_840
```

Values may be proportions or percentages. The script standardises final validation fractions to proportions internally.

---

## Main user parameters

### Requested tasks

| Parameter | Default | Description |
|----------|---------|-------------|
| `calib_trim` | `False` | Run trimming calibration before ASF |
| `asf_only` | `True` | Run only the ASF method; set `False` to run experimental descriptors |
| `generate_d50` | `True` | Generate a D50 map after normalisation |
| `make_post_validation_plots` | `True` | Generate post-validation diagnostic plots if validation files are available |

### Calibration parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `calib_trim_one_unit` | `True` | Treat validation data as one pooled unit in addition to individual files |
| `input_grab_sample_location` | user path | Folder containing validation CSV files |
| `calib_trim_units` | list of filenames | Validation CSV file names |
| `trim_start` | `0` | First α value in percent |
| `trim_end` | `35` | Last α value in percent |
| `trim_step` | `1` | α increment in percent |
| `w1norm_mode` | `'full'` | `'full'` compares full phi-bin fractions; `'percentile'` compares D10, D25, D50, D75, and D90 |

### Directory parameters

| Parameter | Description |
|----------|-------------|
| `input_mat_folder` | Folder containing accumulated flux MAT files |
| `input_grid_mat` | Hydrodynamic grid MAT file |
| `output_ASF_folder` | Final output folder |
| `output_ASF_interim_folder` | Cache folder for per-grain ASF calculations |
| `output_trim_calib_folder` | Folder for calibration reports and figures |
| `output_trim_interim_folder` | Cache folder for calibration distributions and W1norm arrays |
| `input_bss_mat` | Optional bed-shear-stress MAT file for experimental operation 8 |

### ASF parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `max_workers` | `4` | Number of parallel workers for per-grain processing |
| `batch_size` | `2` | Number of grain-size files processed per batch |
| `trim_alpha` | `24` | Trimming proportion α in percent when calibration is off |
| `grain_sizes` | `[75, 105, 150, 210, 300, 420, 600, 840]` | Grain sizes in micrometres; must match MAT filenames |
| `finest_phi` | `4.0` | Finest phi-bin upper boundary |
| `coarsest_phi` | `0.0` | Coarsest phi-bin lower boundary |
| `phi_interval` | `0.5` | Phi-bin interval width |
| `shear_weight_method` | `0` | `0 = none`, `1 = Van Rijn`, `2 = Soulsby` |
| `shear_weight_factor` | `1.0` | Multiplicative factor for critical-shear weighting; if `0`, all weights are 1 |
| `model_type` | `'news'` | Filename suffix used to identify flux MAT files |

---

## Outputs

### Core ASF outputs

| File pattern | Content |
|-------------|---------|
| `ASF_a{alpha}_{grain}um.xyz` | Raw ASF metric per grain-size class |
| `{grain}_perc_{tag}.xyz` | Normalised percentage availability for each grain-size class |
| `{grain}_{tag}_bl.xyz` | Bed-layer fraction for each grain-size class; rows sum to 1 |
| `D50_{tag}_{grains}.xyz` | Median grain-size map generated from phi-scale interpolation |
| `run_config.json` | Complete run-parameter log for reproducibility |

The output tag is constructed as:

```text
{model_type}_a{trim_alpha}
```

For example, if `model_type = 'news'` and `trim_alpha = 24`, the tag is:

```text
news_a24
```

### Calibration outputs

| File pattern | Content |
|-------------|---------|
| `trimming_calibration.jpg` | Per-unit or per-zone W1norm boxplots across α values |
| `trimming_calibration_summary.jpg` | Summary figure with overlaid median W1norm curves |
| `trimming_calibration_report.txt` | Text report listing recommended α values from multiple criteria |

### Post-validation outputs

| File pattern | Content |
|-------------|---------|
| `post_validation_percentile_boxplot_{tag}.jpg` | Observed vs ASF-modelled D10, D25, D50, D75, and D90 boxplots |
| `post_validation_w1norm_heatmap_{tag}.jpg` | Per-percentile W1norm heatmap |
| `post_validation_w1norm_boxplot_{tag}.jpg` | Aggregated pointwise W1norm boxplot by validation unit |

### Cache outputs

| File | Content |
|------|---------|
| `run_params.json` | Parameter subset used to verify cache compatibility |
| `full_run_log.json` | Full record of user-facing parameters |
| `*.npz`, `*.csv`, `*.npy` | Cached per-grain flux calculations, calibration distributions, and W1norm arrays |

---

## Trimming calibration

When `calib_trim = True`, the script calibrates α by comparing ASF-generated distributions against grab-sample validation data.

The calibration workflow reports:

1. **Pooled minimum** — α that minimises pooled median W1norm across validation units.
2. **Kneedle / elbow** — α selected from the elbow of the pooled median curve.
3. **Per-zone or per-unit minima** — α values that minimise W1norm for each validation unit.
4. **Stability plateau** — α range where the pooled median remains within 5% of the minimum.
5. **Normalised composite** — equal-weight min-max normalised multi-unit score.
6. **Pareto-front analysis** — α values that are not dominated across validation units.
7. **BIC segmented regression** — piecewise-linear selection of a plateau onset using a BIC penalty.

After reporting these criteria, the script prompts for a final α. Press Enter to accept the pooled-minimum recommendation, or enter a different integer value.

---

## Validation grain-size harmonisation

The script supports two validation-data cases.

### 1. Exact model-column match

If the validation CSV contains all and only the same `um_*` columns as `grain_sizes`, the data are treated as pre-harmonised model phi-bin fractions. The `um_*` names are interpreted as midpoint labels of the model phi intervals, not as sieve boundaries.

Example:

```text
grain_sizes = [75, 105, 150, 210]
CSV columns = um_75, um_105, um_150, um_210
```

The values are used as provided and normalised to proportions per row.

### 2. Mismatched or additional grain-size columns

If validation columns are missing, additional, or otherwise do not match `grain_sizes`, the script treats the validation data as a raw particle-size distribution. It then:

1. Parses all `um_*` columns.
2. Infers source class edges.
3. Converts observed fractions to a cumulative distribution.
4. Evaluates the cumulative distribution over the model phi-bin boundaries.
5. Re-bins the observed distribution to the model-supported phi intervals.
6. Renormalises the retained mass to sum to 1.
7. Stores the harmonised values under the model midpoint labels.

This avoids directly comparing unmatched observed grain-size supports against model-supported phi classes.

---

## Post-validation diagnostic plots

When `make_post_validation_plots = True`, and validation files are available, the script compares the final ASF distribution to grab-sample observations.

The post-validation routine:

1. Builds a normalised model distribution from the final in-memory ASF dataset.
2. Loads and harmonises validation data.
3. Finds the nearest model grid cell for each validation point.
4. Computes modelled D10, D25, D50, D75, and D90 values.
5. Computes W1norm in the selected mode.
6. Saves three diagnostic plots.

Use this option after calibration or when you want an independent diagnostic check of the final ASF output.

---

## Caching and reproducibility

The script caches expensive intermediate calculations in parameter-verified subdirectories. This allows repeated runs to reuse existing calculations when the relevant parameters have not changed.

Cache directories include a short hash based on parameters such as:

- model type;
- grain sizes;
- input MAT folder;
- grid file;
- trim value;
- phi-bin settings;
- shear-weight settings;
- W1norm mode;
- validation file names.

If parameters change, the script creates a new cache subdirectory rather than overwriting the old cache.

For reproducibility, each final run writes:

```text
run_config.json
```

This file records the complete configuration, final selected α, output tag, elapsed time, and phi intervals.

---

## Critical-shear-stress weighting

The released script includes optional critical-shear-stress weighting:

```python
shear_weight_method = 0  # none
shear_weight_method = 1  # Van Rijn
shear_weight_method = 2  # Soulsby
```

By default:

```python
shear_weight_method = 0
```

so all grain-size weights are equal. Set `shear_weight_method` to `1` or `2` only if the selected modelling design requires compensating for grain-size-dependent mobility in the ASF metric.

---

## Experimental flux descriptors

The ASF method has its own dedicated workflow and does not depend on the experimental operation/analysis codes. These descriptors are retained for research exploration and are only used when:

```python
asf_only = False
```

### Operations: time-series statistic per grain size and flux sign

| `operation_on_fluxes` | Description |
|----------------------|-------------|
| `1` | Arithmetic mean |
| `2` | 95th percentile |
| `3` | Minimum |
| `4` | Maximum |
| `5` | Median |
| `6` | 75th percentile |
| `7` | Exceedance mean above `exceedance_pctile` |
| `8` | Bed-shear-stress-conditioned mean; requires `input_bss_mat` |

### Analyses: combining positive and negative flux components

| `analysis_on_operated_fluxes` | Description |
|------------------------------|-------------|
| `1` | Threshold split: positive flux for grain sizes ≤ `grain_size_thresh`, negative flux above threshold |
| `2` | Positive flux only |
| `3` | Negative flux only, converted to absolute magnitude |
| `4` | Residual depositional flux: positive − negative, clipped at zero |
| `5` | Geometric mean of positive and negative magnitudes |
| `6` | Cumulative flux ratio |

Recommended normal ASF setting:

```python
asf_only = True
```

---

## Troubleshooting

### No MAT files found

Check that:

- `input_mat_folder` points to the correct folder;
- filenames follow `{grain_size}um_{model_type}.mat`;
- `model_type` matches the filename suffix exactly.

### Hydrodynamic grid file not found

Check that `input_grid_mat` is a full valid path. The file must contain `data/X` and `data/Y`.

### Grid and flux shapes do not align

The script can handle equal shapes and +1 offsets in one or both horizontal dimensions. If the mismatch is larger, confirm that the hydrodynamic grid file belongs to the same Delft3D-4 run as the accumulated flux files.

### Validation data fail to load

Check that the CSV contains either `X,Y` or `x,y` coordinate columns, and that grain-size columns are named as `um_{grain_size}`.

### Calibration is slow

Calibration recomputes ASF distributions for every α value unless cached outputs exist. To reduce runtime:

- increase `trim_step`;
- narrow `trim_start` and `trim_end`;
- use cache folders on a fast local disk;
- increase `max_workers` only if memory allows.

### Plot files are missing

Install optional plotting packages:

```bash
pip install matplotlib seaborn
```

Also check that `make_post_validation_plots = True` and that validation CSV files are available.

---

## Citation

If you use this code or ASF-derived outputs, please cite:

```text
Soares, C. C., Persichini, G., Galiforni-Silva, F., Herrling, G., & Winter, C. (2026).
Process-based sediment mapping – an alternative for dynamic and data-scarce coasts.
Applied Ocean Research, 171, 105085.
https://doi.org/10.1016/j.apor.2026.105085
```

BibTeX:

```bibtex
@article{soares2026process,
  title   = {Process-based sediment mapping -- an alternative for dynamic and data-scarce coasts},
  author  = {Soares, Clayton Cyril and Persichini, Gianna and Galiforni-Silva, Filipe and Herrling, Gerald and Winter, Christian},
  journal = {Applied Ocean Research},
  volume  = {171},
  pages   = {105085},
  year    = {2026},
  doi     = {10.1016/j.apor.2026.105085}
}
```

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.
