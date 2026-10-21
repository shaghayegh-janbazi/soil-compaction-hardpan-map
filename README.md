# Interactive Soil Compaction Map from Tillage-Shank Vibration, with Penetrometer Hardpan Depths

**Time-series sensor data · State estimation (Kalman filter + RTS smoother) · Accelerometer–GPS data fusion · Data integrity checks · Experimental validation · Python**

A Python data pipeline that processes **time-series sensor data** recorded on a tractor in the field with the **Agricultural Vibration Data Acquisition Platform (avDAQ)** [1]. It combines vibration from a **tri-axial accelerometer on a tillage shank** (sampled at 102 kHz, summarized as RMS every 0.1 s) with **10 Hz GPS**, and places it next to **cone-penetrometer** measurements. The output is a single interactive HTML map showing:

- **where** soil resistance is higher or lower within the tillage depth (vibration layer, whole field)
- **at what depth** a hardpan starts at each measured point (penetrometer layer)

<!-- DROP map_overview.png HERE (delete this line after the image link appears) -->

*Screenshot of the generated map (2022 field season): vibration grid (green = low, red = high median RMS), penetrometer points labeled with hardpan top depth, and the popup of one insertion with its cone-index-vs-depth profile. The interactive HTML file is not published because it embeds the field data.*

---

## Skills applied

- **Time-series sensor data processing:** accelerometer RMS and GPS streams aligned on a common 0.1 s time base and processed in a reproducible Python pipeline.
- **State estimation:** a constant-velocity **Kalman filter** with a **Rauch–Tung–Striebel smoother** recovers a continuous trajectory from noisy GPS with repeated fixes.
- **Data integrity:** automatic detection of repeated GPS fixes, data gaps, unit checks and inconsistent column names.
- **Experimental validation:** sensor-derived values compared with independent ground-truth measurements (cone penetrometer).
- **Geospatial analysis and visualization:** robust grid statistics and an interactive map built with MapLibre GL JS.

---

## 1. Background

A cone penetrometer gives a detailed strength-vs-depth profile at one point per insertion. A tillage shank is in the soil across the whole field. Our earlier published work showed that **cultivator shank vibration, together with soil moisture, can be used to predict soil compaction** [2]. This project turns field-recorded shank vibration into a **field-scale map** and shows it together with penetrometer measurements.

The two layers complement each other:

| Layer | Sensor | What it shows | Coverage |
|---|---|---|---|
| Vibration grid | Accelerometer on the shank, axis B (x-axis, horizontal) | Relative soil resistance within the tillage depth (4 in) | Whole field |
| Penetrometer points | Cone penetrometer | Cone index vs depth, hardpan top and bottom (≥ 300 psi) | Measured points |

The vibration layer describes the soil the shank works in; the penetrometer layer adds the depth information below it.

---

## 2. Pipeline

<!-- DROP pipeline.png HERE (delete this line after the image link appears) -->

The vibration/GPS branch (Steps 1–4) and the penetrometer branch (Step 5) are processed independently and meet at validation (Step 6).

---

## 3. Methods

### 3.1 Load and data-integrity checks
- Reads comma- or tab-separated files automatically and normalizes column names (`lon`/`longitude` → `long`, `latitude` → `lat`).
- Converts the required columns to numeric; invalid entries become missing values.
- Reports the number of rows and the recording duration (rows ÷ 10 Hz) to confirm that the vibration–GPS merge is complete.
- Detects **repeated GPS fixes**: a fix identical to the previous one (same latitude and longitude) is treated as a missing measurement and later replaced by the smoothed estimate.
- Detects and corrects the longitude sign in the penetrometer file when it differs from the GPS file.

### 3.2 GPS state estimation: Kalman filter and RTS smoother

Each vibration value is placed on the map by its GPS position, so position quality directly affects map quality. Positions are estimated with a **linear Kalman filter** and refined with a **Rauch–Tung–Striebel (RTS) smoother**.

**Coordinates.** Latitude/longitude are converted to a local east–north frame in meters (equirectangular projection centered on the field).

**State and motion model** (constant velocity, Δt = 0.1 s):

$$
\mathbf{x}_k = \begin{bmatrix} p_x & p_y & v_x & v_y \end{bmatrix}^T ,
\qquad
\mathbf{x}_{k+1} = \mathbf{F}\,\mathbf{x}_k + \mathbf{w}_k ,
\qquad
\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

**Measurement model** (GPS measures position):

$$
\mathbf{z}_k = \mathbf{H}\,\mathbf{x}_k + \mathbf{v}_k ,
\qquad
\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix},
\qquad
\mathbf{R} = \sigma_{GPS}^2\,\mathbf{I}
$$

**Noise settings**
- **Process noise Q:** discrete white-noise-acceleration model, σₐ = 0.5 m/s², matching the slow, smooth motion of a tractor.
- **Measurement noise R:** the pipeline estimates σ_GPS from parked segments (speed < 0.3 mph), where any position change is sensor noise. When parked segments do not contain enough distinct fixes, a configurable value is used. For the 2022 recording, σ_GPS = 0.5 m.

**Filter and smoother**
1. **Forward pass (Kalman filter):** predicts each position from the motion model and corrects it with the GPS fix, weighted by the Kalman gain. Where a fix is missing or repeated, the filter uses the prediction.
2. **Backward pass (RTS smoother):** since the data are processed offline, each estimate is refined with both past and future measurements, giving a smoother and more accurate trajectory and filling repeated-fix rows with consistent positions.

**Robustness**
- **Gap handling:** the track is split into independent segments where a jump exceeds 5 m plus 4 m/s × elapsed time.
- **Extrapolation limit:** at most 1 s (10 samples) beyond the last valid fix of a segment.
- **Order of operations:** smoothing runs on the complete 0.1 s time series before any rows are removed, so the motion model always sees a continuous time base.
- **Speed-unit check:** the ratio of the logged speed column to the smoothed speed (m/s) identifies the unit of the speed column (m/s, km/h or mph).

### 3.3 Operating-state filter
Rows are kept for mapping when speed > 3 mph (steady tillage, chosen from the speed distribution of the recording), RMS is valid, and a smoothed position exists. The filter is applied after smoothing.

### 3.4 Spatial statistics
- Samples are binned into a **3 m × 3 m grid** using the smoothed positions.
- Per cell: **median** RMS (robust to single impulsive events such as stone strikes), **25th/75th percentiles**, **sample count** and median speed.
- Cells with fewer than 5 samples are not shown.
- Cells are colored in 8 classes whose limits are the 2nd–98th percentiles of the cell medians (fixed limits can be set instead, e.g. to compare fields or years).

### 3.5 Penetrometer processing
- Detects all depth columns automatically (`Interval 1` … `Interval N`, including variants such as `Interval-16`); the depth of each reading is the interval index × the configured depth step.
- Converts cone index **kPa → psi** (÷ 6.894757).
- Per insertion:
  - **hardpan top:** first depth ≥ 300 psi (≈ 2.07 MPa)
  - **hardpan bottom:** last depth of that continuous layer, flagged when it continues to the deepest reading
  - **maximum cone index** and its depth
  - **cone index at the tillage depth**, by linear interpolation
- Optional masking of readings below the logged maximum depth.

### 3.6 Experimental validation
For each penetrometer point, the pipeline takes the **median RMS of all tilling samples within 5 m** and compares it with **cone index at the tillage depth** using **Spearman rank correlation**, which measures monotonic association without assuming a linear relationship. It also reports median vibration for points with and without a hardpan. The comparison runs when at least 5 points have nearby vibration data, and the per-point results are written to a CSV file.

### 3.7 Interactive map
One self-contained HTML file (MapLibre GL JS). All processed data are embedded in the file as JSON/GeoJSON; the map library, fonts, satellite imagery (Esri) and elevation tiles load online.
- satellite basemap with **3-D terrain** and adjustable height exaggeration
- toggleable layers: vibration grid · **Kalman-smoothed path** · **raw GPS fixes** · penetrometer points (path and raw GPS show every 2nd sample to keep the file small)
- penetrometer points colored by result (hardpan / no hardpan / no data) and labeled with hardpan top depth
- click a cell → median, 25th–75th percentile range, sample count, median speed
- click a point → hardpan depth range, maximum cone index and its depth, cone index at tillage depth, nearby vibration, and an **inline cone-index-vs-depth chart** with the 300 psi threshold and the tillage-depth line

---

## 4. Processing summary (2022 field season)

| Step | Result |
|---|---|
| Recording | 75,626 samples (126 min at 10 Hz) |
| Repeated GPS fixes replaced by smoothed estimates | 24,083 (31.8 %) |
| Kalman/RTS position correction | median 0.13 m, 95th percentile 0.38 m |
| Continuous track segments | 1 |
| Speed unit (verified from smoothed trajectory) | mph |
| Samples kept for mapping (speed > 3 mph) | 56,222 (74.3 %) |
| Grid cells mapped (≥ 5 samples) | 3,170 |
| Penetrometer insertions with nearby vibration data | 274 of 285 |

---

## 5. Repository structure

```text
soil-compaction-hardpan-map/
├── compaction_hardpan_map_2022.py   # full pipeline (Steps 1–7)
├── requirements.txt                 # numpy, pandas
└── README.md
```

## 6. How to run

```bash
pip install -r requirements.txt        # numpy, pandas
python compaction_hardpan_map_2022.py <vibration_gps.csv> <penetrometer.csv> <output_dir>
```

**Input 1: vibration + GPS** (one row = 0.1 s)

| lat | long | speed (mph) | RMS_ACC_A | RMS_ACC_B | RMS_ACC_C |
|---|---|---|---|---|---|

The pipeline uses `lat`, `long`, `speed` and `RMS_ACC_B`.

**Input 2: penetrometer** (one row = one insertion): `Decimal_Latitude`, `Decimal_Longitude`, `Interval 1 … Interval N` (cone index in kPa), plus logger metadata.

**Outputs**

| File | Content |
|---|---|
| `compaction_hardpan_2022.html` | Interactive map |
| `rmsB_grid_cells_2022.csv` | Per cell: center coordinates, median, 25th/75th percentiles, count, median speed, color class |
| `pen_vs_vibration_2022.csv` | Per insertion: location, hardpan top/bottom, maximum cone index and its depth, cone index at tillage depth, nearby vibration median and sample count |

All parameters (speed threshold, grid size, hardpan threshold, depth step, tillage depth, search radius, Kalman noise settings) are in the configuration block at the top of the script.

---

## 7. Scope

- The vibration input is RMS per 0.1 s; the pipeline works at this level and does not process the raw 102 kHz waveform.
- The vibration layer describes soil resistance within the tillage depth; deeper layers are described by the penetrometer.
- Each row of the merged file represents 0.1 s (10 Hz).

---

## 8. Data availability
Field data were collected at Kansas State University (Biological and Agricultural Engineering) with the avDAQ platform [1]. The data and the generated HTML map (which embeds the data) are not included in this repository.

## 9. References
1. Ridder, E., Brokesh, E., JanbaziAlamdari, S., & Vail, B. (2023). Development of the Agricultural Vibration Data Acquisition Platform (avDAQ). In *2023 ASABE Annual International Meeting* (p. 1). American Society of Agricultural and Biological Engineers.
2. Janbazialamdari, S., Flippo, D., Ridder, E., & Brokesh, E. (2025). Soil compaction prediction in precision agriculture using cultivator shank vibration and soil moisture data. *Agriculture, 15*(17), 1896. https://www.mdpi.com/2077-0472/15/17/1896

## Author
**Shaghayegh Janbazialamdari**, PhD candidate, Biological and Agricultural Engineering, Kansas State University
[GitHub](https://github.com/shaghayegh-janbazi) · [LinkedIn](https://linkedin.com/in/shaghayegh1)
