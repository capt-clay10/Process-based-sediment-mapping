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

