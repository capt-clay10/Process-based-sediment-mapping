## Overview
This script post-processes **Delft3D-4** accumulated seabed flux output to generate seabed sediment distribution maps. The main workflow implements the **Central accumulated sediment flux (ASF)** approach: it compresses the flux time series per grain size, combines positive (depositional) and negative (erosional) fluxes, and converts the result into relative grain-size availability and bed-layer fractions.

## What the code does
- Reads accumulated flux MAT files for multiple grain-size classes and a hydrodynamic grid MAT file (HDF5 / v7.3 format).
- Separates positive and negative fluxes and applies a chosen statistical operator (e.g. mean, percentiles, trimmed mean / ASF).
- Combines positive and negative components according to a selected analysis method (ASF is the default).
- Optionally applies simple shear-stress-based weighting between grain-size classes.
- Normalises the combined flux per cell across grain sizes to obtain:
  - percentage contribution of each size class, and
  - bed-layer fractions suitable as input for morphodynamic models.
- Optionally computes a **D50** map from the ASF-based grain-size percentages.
- Writes all outputs as `.xyz` text files (`X Y Z`) that can be imported into GIS or plotting software.

## Possible exploratory combinations possible
### Operation (`op`) options – time-series statistic per grain size and flux sign (positive = deposition and negative = erosion) for each grid cell

| `op` | Name                       | Description (per grid cell, per grain, per sign)                                         | Extra input/notes                                     |
|------|----------------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------|
| 1    | Mean                       | NaN-aware arithmetic mean of the flux time series                                        | Works for all analyses 1–7                             |
| 2    | 95th percentile            | 95th percentile of flux values                                                           |                                                         |
| 3    | Minimum                    | Minimum flux value                                                                        |                                                         |
| 4    | Maximum                    | Maximum flux value                                                                        |                                                         |
| 5    | Median                     | Median flux value                                                                         |                                                         |
| 6    | Trimmed mean (ASF core)    | Mean over the interquartile range (25–75 %); returns both trimmed mean and counts `N_trim` | Needed for ASF when combined with `analysis = 6`       |
| 7    | 75th percentile            | 75th percentile of flux values                                                           |                                                         |
| 8    | Exceedance mean            | Mean of values above the `exceed_pct` percentile                                         | Uses `exceed_pct` (default 25 %)                       |
| 9    | BSS-conditioned mean       | Mean of flux values at times when BSS exceeds its `exceed_pct` percentile                | Requires BSS file with matching time dimension         |

> For `analysis = 7` (CFR), the same `op` codes are applied to positive/negative flux **magnitudes** before forming the cumulative flux ratio.


### Analysis options – combination of positive/negative fluxes at each grid cell

| `analysis` | Name                         | Description (per grid cell, per grain)                                                                    | Special requirements / comments                                   |
|------------|------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 1          | Threshold split              | Uses positive statistic for grain sizes ≤ `thresh` and negative statistic for grain sizes > `thresh`      | Requires `thresh` and grain sizes on both sides of the threshold  |
| 2          | Positive only                | Uses only the positive (depositional) statistic                                                           |                                                                   |
| 3          | Negative only                | Uses only the negative (erosional, absolute value) statistic                                              |                                                                   |
| 4          | Residual depositional flux   | Positive − negative; negative values clipped to 0 (emphasises net deposition)                             | Requires both pos and neg stats (analyses 1–6)                    |
| 5          | Geometric-mean transport     | √(pos · neg), with small epsilon; emphasises co-occurrence of deposition and erosion                      | Requires both pos and neg stats                                  |
| 6          | Total transport / ASF        | If `op = 6` and `N_trim` available: ASF = μ⁺·N⁺ + μ⁻·N⁻; otherwise total = |pos| + |neg|                  | ASF configuration is `op = 6`, `analysis = 6`                     |
| 7          | Cumulative flux ratio (CFR)  | Ratio of positive to negative statistics minus 1, i.e. CFR − 1, computed from sign-separated flux fields | Uses only CFR branch in `operate_mat` (choice = 3)                |


### Recommended combinations

| Purpose                             | Recommended setting                  | Comment                                             |
|-------------------------------------|--------------------------------------|-----------------------------------------------------|
| ASF sediment distribution (default) | `op = 6`, `analysis = 6`            | Central, frequency-weighted gross exchange per grain|
| Simple depositional intensity       | `op = 1,2,5,7`, `analysis = 2`      | Positive flux only                                  |
| Simple erosional intensity          | `op = 1,2,5,7`, `analysis = 3`      | Negative flux only (absolute values)                |
| Net residual deposition             | `op = 1 or 6`, `analysis = 4`       | Highlights areas of persistent deposition           |
| CFR diagnostics                     | `op = 1–9`, `analysis = 7`          | Different `op` give different CFR summaries         |


