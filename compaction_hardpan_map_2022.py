import os
import sys
import re
import json
import math
import numpy as np
import pandas as pd

ACC_CSV = os.path.join("data", "merged_acc_gps2022.csv")          # default input paths;
PEN_CSV = os.path.join("data", "penData_correct_latlong2022.csv")  # override with command-line arguments
OUT_DIR = os.path.dirname(ACC_CSV)
if len(sys.argv) >= 3:
    ACC_CSV, PEN_CSV = sys.argv[1], sys.argv[2]
    OUT_DIR = sys.argv[3] if len(sys.argv) >= 4 else os.path.dirname(os.path.abspath(ACC_CSV))
OUTPUT_HTML    = os.path.join(OUT_DIR, "compaction_hardpan_2022.html")
VALIDATION_CSV = os.path.join(OUT_DIR, "pen_vs_vibration_2022.csv")
CELLS_CSV      = os.path.join(OUT_DIR, "rmsB_grid_cells_2022.csv")

GPS_RATE_HZ = 10.0          
SPEED_MIN   = 3.0           
RMS_COL     = "RMS_ACC_B"

#Kalman (RTS) smoother
KF_SIGMA_ACC        = 0.5   
KF_GPS_SIGMA_DEFAULT = 0.5  
KF_PARKED_SPEED     = 0.3   
KF_JUMP_M           = 5.0   
KF_MAX_SPEED_MS     = 4.0   
KF_MAX_EXTRAP_ROWS  = 10    
CELL_M            = 3.0     
MIN_ROWS_PER_CELL = 5       
KPA_PER_PSI      = 6.894757 # 1 psi = 6.894757 kPa
HARDPAN_PSI      = 300.0    # >= 300 psi (~2.07 MPa) = hardpan
DEPTH_STEP_IN    = 1.0      
USE_MAXDEPTH_MASK = False   
SHANK_DEPTH_IN   = 4     
PEN_RADIUS_M     = 5.0      
RMS_VALUE_STOPS = None
CLASS_COLORS = [
    (26, 150, 65), (102, 189, 99), (166, 217, 106), (254, 224, 139),
    (253, 174, 97), (244, 109, 67), (215, 48, 39), (165, 0, 38),
]
TERRAIN_EXAGGERATION = 8
DISPLAY_STEP = 2            # draw every 2nd GPS point for path/raw layers (keeps HTML small)
M_PER_DEG = 111320.0

def read_table(path):
    if not os.path.exists(path):
        sys.exit(f"ERROR: cannot find '{path}'")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")  # comma OR tab
    df.columns = [str(c).strip() for c in df.columns]
    return df

class LocalProj:
    """Lat/lon <-> local metres (x east, y north). Accurate to cm over a field."""
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.k = math.cos(math.radians(lat0))

    def fwd(self, lat, lon):
        return (lon - self.lon0) * M_PER_DEG * self.k, (lat - self.lat0) * M_PER_DEG

    def inv(self, x, y):
        return self.lat0 + y / M_PER_DEG, self.lon0 + x / (M_PER_DEG * self.k)

def num(v, nd=4):
    """JSON-safe number (NaN -> None)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None

def rts_smooth(zx, zy, has_meas, dt, sigma_acc, sigma_gps):
    """
    Constant-velocity Kalman filter (forward) + Rauch-Tung-Striebel smoother (backward).
    State = [x, y, vx, vy].  Rows without a valid GPS fix (has_meas=False) get
    prediction only; the backward pass then fills them using future fixes too.
    """
    n = len(zx)
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    q = sigma_acc ** 2
    Q = q * np.array([[dt**4/4, 0, dt**3/2, 0],
                      [0, dt**4/4, 0, dt**3/2],
                      [dt**3/2, 0, dt**2, 0],
                      [0, dt**3/2, 0, dt**2]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    R = np.eye(2) * sigma_gps ** 2
    I4 = np.eye(4)

    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))     
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))     

    first = int(np.flatnonzero(has_meas)[0])
    x = np.array([zx[first], zy[first], 0.0, 0.0])
    P = np.diag([sigma_gps**2, sigma_gps**2, 4.0, 4.0])  

    for k in range(first, n):
        if k > first:
            x = F @ x
            P = F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if has_meas[k]:
            innov = np.array([zx[k], zy[k]]) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ innov
            P = (I4 - K @ H) @ P
        xf[k], Pf[k] = x, P

    xs = xf.copy(); Ps = Pf.copy()
    for k in range(n - 2, first - 1, -1):
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    xs[:first] = np.nan
    return xs


def estimate_gps_sigma(x, y, speed, meas):
    """GPS noise (m) = position scatter while the tractor is parked."""
    parked = (speed < KF_PARKED_SPEED) & meas
    idx = np.flatnonzero(parked)
    if len(idx) < 20:
        return None
    blocks = np.split(idx, np.flatnonzero(np.diff(idx) > 10) + 1)
    resid = []
    for b in blocks:
        if len(b) >= 20:
            resid += [x[b] - x[b].mean(), y[b] - y[b].mean()]
    if not resid:
        return None
    return float(np.std(np.concatenate(resid)))

print(f"\n[1] Reading {ACC_CSV}")
acc = read_table(ACC_CSV)
acc = acc.rename(columns={"lon": "long", "longitude": "long", "latitude": "lat"})
for c in ["lat", "long", "speed", RMS_COL]:
    if c not in acc.columns:
        sys.exit(f"ERROR: column '{c}' not found. Columns are: {list(acc.columns)}")
    acc[c] = pd.to_numeric(acc[c], errors="coerce")
n = len(acc)
print(f"    rows: {n:,}  ->  {n / GPS_RATE_HZ / 60:.1f} min at {GPS_RATE_HZ:g} Hz")

lat = acc["lat"].to_numpy(float)
lon = acc["long"].to_numpy(float)
spd = acc["speed"].to_numpy(float)
rmsb = acc[RMS_COL].to_numpy(float)
print("\n[2] RTS Kalman smoother on GPS")
valid = np.isfinite(lat) & np.isfinite(lon)
frozen = np.zeros(n, bool)
frozen[1:] = (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1])   
meas = valid & ~frozen
print(f"    valid fixes: {valid.sum():,}   frozen/repeated: {(valid & frozen).sum():,} "
      f"({100 * (valid & frozen).sum() / max(valid.sum(), 1):.1f} %) -> treated as missing")

proj = LocalProj(float(np.nanmedian(lat[valid])), float(np.nanmedian(lon[valid])))
zx, zy = proj.fwd(lat, lon)

sigma_gps = estimate_gps_sigma(zx, zy, spd, meas)
if sigma_gps is None or not (0.01 <= sigma_gps <= 5):
    print(f"    GPS noise could not be estimated from parked rows -> default {KF_GPS_SIGMA_DEFAULT} m")
    sigma_gps = KF_GPS_SIGMA_DEFAULT
else:
    print(f"    GPS noise estimated from parked rows: {sigma_gps:.3f} m")
sigma_gps = max(sigma_gps, 0.02)
dt = 1.0 / GPS_RATE_HZ
mi = np.flatnonzero(meas)
jump = np.hypot(np.diff(zx[mi]), np.diff(zy[mi]))
allowed = KF_JUMP_M + KF_MAX_SPEED_MS * np.diff(mi) * dt
seg_starts = [0] + list(mi[1:][jump > allowed]) + [n]
print(f"    continuous segments: {len(seg_starts) - 1}")

sx = np.full(n, np.nan); sy = np.full(n, np.nan)
svx = np.full(n, np.nan); svy = np.full(n, np.nan)
for a, b in zip(seg_starts[:-1], seg_starts[1:]):
    m = meas[a:b]
    if m.sum() < 2:
        continue
    xs = rts_smooth(zx[a:b], zy[a:b], m, dt, KF_SIGMA_ACC, sigma_gps)
    last = int(np.flatnonzero(m)[-1])
    xs[last + KF_MAX_EXTRAP_ROWS + 1:] = np.nan         
    sx[a:b], sy[a:b], svx[a:b], svy[a:b] = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3]

slat, slon = proj.inv(sx, sy)
corr_m = np.hypot(sx - zx, sy - zy)[meas]
print(f"    position correction: median {np.nanmedian(corr_m):.2f} m, "
      f"95th pct {np.nanpercentile(corr_m, 95):.2f} m")
kf_speed = np.hypot(svx, svy)
moving = (kf_speed > 0.8) & np.isfinite(spd) & (spd > 0)
if moving.sum() > 50:
    ratio = float(np.median(spd[moving] / kf_speed[moving]))
    guess = min({"m/s": 1.0, "km/h": 3.6, "mph": 2.237}.items(), key=lambda kv: abs(kv[1] - ratio))[0]
    print(f"    speed column / Kalman speed (m/s) = {ratio:.2f}  ->  'speed' looks like {guess}")
print(f"\n[3] Filtering: speed > {SPEED_MIN}")
work = (spd > SPEED_MIN) & np.isfinite(rmsb) & np.isfinite(sx)
print(f"    kept {work.sum():,} of {n:,} rows ({100 * work.sum() / n:.1f} %)")
wx, wy, wr, ws = sx[work], sy[work], rmsb[work], spd[work]
if work.sum() == 0:
    sys.exit("ERROR: no rows left after the speed filter.")
print(f"\n[4] Gridding RMS_ACC_B into {CELL_M:g} m cells (median per cell)")
g = pd.DataFrame({"gx": np.floor(wx / CELL_M).astype(int),
                  "gy": np.floor(wy / CELL_M).astype(int),
                  "rms": wr, "spd": ws})
cells = (g.groupby(["gx", "gy"])
          .agg(med=("rms", "median"), q25=("rms", lambda v: v.quantile(.25)),
               q75=("rms", lambda v: v.quantile(.75)), n=("rms", "size"), spd=("spd", "median"))
          .reset_index())
n_all = len(cells)
cells = cells[cells["n"] >= MIN_ROWS_PER_CELL].reset_index(drop=True)
print(f"    cells: {n_all:,}  ->  {len(cells):,} with >= {MIN_ROWS_PER_CELL} rows")

if RMS_VALUE_STOPS is None:
    stops = np.percentile(cells["med"], np.linspace(2, 98, 9))
    stops = np.maximum.accumulate(stops)
    stops = stops + np.arange(9) * 1e-9           
    stops_src = "automatic (2nd-98th percentile of cell medians)"
else:
    stops = np.asarray(RMS_VALUE_STOPS, float)
    stops_src = "absolute (from config)"
print(f"    class limits, {stops_src}: {np.round(stops, 3).tolist()}")
cls = np.clip(np.searchsorted(stops, cells["med"].to_numpy(), side="right") - 1, 0, 7)
cells["cls"] = cls

cell_feats = []
for r in cells.itertuples():
    x0, y0 = r.gx * CELL_M, r.gy * CELL_M
    ring = [(x0, y0), (x0 + CELL_M, y0), (x0 + CELL_M, y0 + CELL_M), (x0, y0 + CELL_M), (x0, y0)]
    coords = [[round(proj.inv(px, py)[1], 7), round(proj.inv(px, py)[0], 7)] for px, py in ring]
    rr, gg, bb = CLASS_COLORS[int(r.cls)]
    cell_feats.append({"type": "Feature",
                       "geometry": {"type": "Polygon", "coordinates": [coords]},
                       "properties": {"med": num(r.med), "q25": num(r.q25), "q75": num(r.q75),
                                      "n": int(r.n), "spd": num(r.spd, 2),
                                      "color": f"rgb({rr},{gg},{bb})"}})
cells_out = cells.copy()
cells_out["center_lat"], cells_out["center_lon"] = proj.inv((cells.gx + .5) * CELL_M, (cells.gy + .5) * CELL_M)
cells_out.to_csv(CELLS_CSV, index=False)
print(f"\n[5] Reading penetrometer {PEN_CSV}")
pen = read_table(PEN_CSV)
int_cols = {}
for c in pen.columns:
    m = re.fullmatch(r"Interval[\s\-_]*(\d+)", c, flags=re.IGNORECASE)
    if m:
        int_cols[int(m.group(1))] = c
if not int_cols:
    sys.exit("ERROR: no 'Interval N' columns found in the penetrometer file.")
ks = sorted(int_cols)
depth_in = np.array(ks, float) * DEPTH_STEP_IN
print(f"    depth columns: Interval {ks[0]} .. Interval {ks[-1]}  ({len(ks)} readings)"
      f"  -> depths {depth_in[0]:g} .. {depth_in[-1]:g} in")
if "Interval" in pen.columns:
    print(f"    'Interval' column values (depth step?): {sorted(pd.unique(pen['Interval'].dropna()))[:5]}")
if "MaxDepth" in pen.columns:
    md_vals = pd.to_numeric(pen["MaxDepth"], errors="coerce")
    print(f"    'MaxDepth' range: {md_vals.min()} .. {md_vals.max()}  (mask applied: {USE_MAXDEPTH_MASK})")

kpa = pen[[int_cols[k] for k in ks]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
if USE_MAXDEPTH_MASK and "MaxDepth" in pen.columns:
    md = pd.to_numeric(pen["MaxDepth"], errors="coerce").to_numpy(float)
    for i in range(len(pen)):
        if np.isfinite(md[i]) and md[i] > 0:
            kpa[i, depth_in > md[i]] = np.nan
psi = kpa / KPA_PER_PSI
print(f"    converted kPa -> psi (/ {KPA_PER_PSI});  {HARDPAN_PSI:g} psi = "
      f"{HARDPAN_PSI * KPA_PER_PSI:.0f} kPa = {HARDPAN_PSI * KPA_PER_PSI / 1000:.2f} MPa")

plat = pd.to_numeric(pen["Decimal_Latitude"], errors="coerce").to_numpy(float)
plon = pd.to_numeric(pen["Decimal_Longitude"], errors="coerce").to_numpy(float)
if np.nanmedian(plon) > 0 > np.nanmedian(lon):
    print("    WARNING: penetrometer longitudes are positive, vibration longitudes negative -> flipping sign")
    plon = -plon
px, py = proj.fwd(plat, plon)

group_col = "Group" if "Group" in pen.columns else None
ins_col = "Insert" if "Insert" in pen.columns else None
pen_rows, pen_feats, pen_profiles = [], [], {}
for i in range(len(pen)):
    p = psi[i]
    ok = np.isfinite(p)
    grp = str(pen[group_col].iloc[i]) if group_col else ""
    ins = str(pen[ins_col].iloc[i]) if ins_col else str(i + 1)
    pid = f"{grp}-{ins}" if grp else ins

    hp_top = hp_bot = max_psi = d_max = psi_shank = np.nan
    hp_to_end = False
    if ok.any():
        max_psi = float(np.nanmax(p)); d_max = float(depth_in[np.nanargmax(p)])
        above = ok & (p >= HARDPAN_PSI)
        if above.any():
            j0 = int(np.flatnonzero(above)[0]); j1 = j0
            while j1 + 1 < len(p) and above[j1 + 1]:
                j1 += 1
            hp_top, hp_bot = float(depth_in[j0]), float(depth_in[j1])
            hp_to_end = j1 == int(np.flatnonzero(ok)[-1])     
        if SHANK_DEPTH_IN is not None and ok.sum() >= 2:
            psi_shank = float(np.interp(SHANK_DEPTH_IN, depth_in[ok], p[ok]))

    near_med = np.nan; near_n = 0
    if np.isfinite(px[i]):
        dd = np.hypot(wx - px[i], wy - py[i])
        sel = dd <= PEN_RADIUS_M
        near_n = int(sel.sum())
        if near_n:
            near_med = float(np.median(wr[sel]))

    status = "hardpan" if np.isfinite(hp_top) else ("none" if ok.any() else "nodata")
    lbl = (f'{hp_top:g}"' if status == "hardpan" else ("no HP" if status == "none" else "?"))
    pen_rows.append({"id": pid, "group": grp, "insert": ins, "lat": plat[i], "lon": plon[i],
                     "hardpan_top_in": hp_top, "hardpan_bottom_in": hp_bot,
                     "hardpan_to_max_depth": hp_to_end, "max_psi": max_psi,
                     "depth_of_max_in": d_max, "psi_at_shank_depth": psi_shank,
                     f"rmsB_median_{PEN_RADIUS_M:g}m": near_med, "rmsB_rows_used": near_n})
    if np.isfinite(plat[i]) and np.isfinite(plon[i]):
        pen_feats.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [round(plon[i], 7), round(plat[i], 7)]},
                          "properties": {"id": pid, "status": status, "lbl": lbl,
                                         "hp_top": num(hp_top, 1), "hp_bot": num(hp_bot, 1),
                                         "hp_end": bool(hp_to_end), "max_psi": num(max_psi, 0),
                                         "d_max": num(d_max, 1), "psi_shank": num(psi_shank, 0),
                                         "rms": num(near_med), "rms_n": near_n}})
        pen_profiles[pid] = {"d": [num(v, 2) for v in depth_in], "p": [num(v, 1) for v in p]}

val = pd.DataFrame(pen_rows)
n_hp = int(val["hardpan_top_in"].notna().sum())
print(f"    insertions: {len(val)}   with hardpan (>= {HARDPAN_PSI:g} psi): {n_hp}   "
      f"without: {len(val) - n_hp}")
if n_hp:
    print(f"    hardpan top depth: median {val['hardpan_top_in'].median():g} in, "
          f"range {val['hardpan_top_in'].min():g} .. {val['hardpan_top_in'].max():g} in")

print(f"\n[6] Validation (vibration within {PEN_RADIUS_M:g} m of each penetrometer point)")
rcol = f"rmsB_median_{PEN_RADIUS_M:g}m"
v = val[val[rcol].notna()]
print(f"    points with nearby vibration: {len(v)} of {len(val)}")
if len(v) >= 5:
    target = "psi_at_shank_depth" if SHANK_DEPTH_IN is not None else "max_psi"
    rho = v[[rcol, target]].corr(method="spearman").iloc[0, 1]
    print(f"    Spearman rho (RMS_ACC_B vs {target}) = {rho:.2f}   (n = {len(v)})")
    a = v.loc[v["hardpan_top_in"].notna(), rcol]; b = v.loc[v["hardpan_top_in"].isna(), rcol]
    if len(a) and len(b):
        print(f"    median RMS_ACC_B   hardpan points: {a.median():.3f} (n={len(a)})   "
              f"no-hardpan points: {b.median():.3f} (n={len(b)})")
    if SHANK_DEPTH_IN is None:
        print("    NOTE: set SHANK_DEPTH_IN to compare with psi at the depth the shank actually works.")
else:
    print("    too few points for a correlation.")
val.to_csv(VALIDATION_CSV, index=False)


print("\n[7] Building HTML map")
path_lines = []
for a, b in zip(seg_starts[:-1], seg_starts[1:]):
    idx = np.arange(a, b, DISPLAY_STEP)
    idx = idx[np.isfinite(slat[idx])]
    if len(idx) >= 2:
        path_lines.append([[round(slon[k], 7), round(slat[k], 7)] for k in idx])
raw_idx = np.flatnonzero(meas)[::DISPLAY_STEP]
raw_pts = [[round(lon[k], 7), round(lat[k], 7)] for k in raw_idx]

legend_steps = "".join(f'<div style="background:rgb({r},{g_},{b})"></div>' for r, g_, b in CLASS_COLORS)
legend_labels = "".join(f"<span>{s:.3g}</span>" for s in stops)
c_lat, c_lon = proj.inv(np.nanmean(wx), np.nanmean(wy))

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Compaction & Hardpans 2022</title>
<link href="https://unpkg.com/maplibre-gl@4.5.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
  *{margin:0;padding:0;box-sizing:border-box;}
  body{font-family:'DM Sans',sans-serif;background:#0f1117;color:#e0e0e6;height:100vh;overflow:hidden;}
  #map{position:absolute;inset:0;}
  #toolbar{position:absolute;top:16px;left:16px;z-index:10;display:flex;gap:8px;flex-wrap:wrap;max-width:calc(100% - 300px);}
  .tool-btn{background:rgba(15,17,23,0.92);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);color:#c0c0cc;
    padding:8px 14px;border-radius:8px;cursor:pointer;font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
    transition:all .2s;user-select:none;}
  .tool-btn:hover{background:rgba(30,33,45,.95);color:#fff;}
  .tool-btn.active{background:rgba(56,139,253,.2);color:#58a6ff;border-color:rgba(88,166,255,.4);}
  #exag-panel{position:absolute;top:68px;left:16px;z-index:10;background:rgba(15,17,23,.92);border:1px solid rgba(255,255,255,.08);
    border-radius:8px;padding:10px 14px;display:none;min-width:220px;}
  #exag-panel label{font-family:'JetBrains Mono',monospace;font-size:11px;color:#8b8fa3;text-transform:uppercase;display:flex;justify-content:space-between;margin-bottom:6px;}
  #exag-slider{width:100%;accent-color:#58a6ff;}
  .panel{position:absolute;z-index:10;background:rgba(15,17,23,.92);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.08);
    border-radius:10px;padding:14px 16px;}
  .panel h4{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;color:#8b8fa3;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;}
  #legend{left:16px;bottom:30px;min-width:300px;}
  #pen-legend{left:16px;bottom:140px;min-width:220px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#c0c0cc;line-height:1.9;}
  #info{right:16px;top:16px;width:270px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#a0a4b8;line-height:1.7;}
  #info b{color:#e0e0e6;}
  .leg-steps{display:flex;height:14px;border-radius:7px;overflow:hidden;margin-bottom:6px;}
  .leg-steps div{flex:1;}
  .leg-labels{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10px;color:#6e7286;}
  .dot{display:inline-block;width:11px;height:11px;border-radius:50%;border:2px solid #0f1117;vertical-align:middle;margin-right:6px;}
  .maplibregl-popup-content{background:rgba(15,17,23,.96)!important;color:#e0e0e6!important;border:1px solid rgba(255,255,255,.1)!important;
    border-radius:10px!important;padding:12px 16px!important;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.7;}
  .maplibregl-popup-tip{border-top-color:rgba(15,17,23,.96)!important;}
  .maplibregl-ctrl-group{background:rgba(15,17,23,.92)!important;}
  .maplibregl-ctrl button .maplibregl-ctrl-icon{filter:invert(1) opacity(.6);}
  .pt{font-family:'DM Sans',sans-serif;font-weight:700;font-size:13px;color:#58a6ff;margin-bottom:4px;}
  .pv{color:#e0e0e6;font-weight:600;} .ph{color:#f85149;font-weight:700;} .pg{color:#3fb950;font-weight:700;}
  #loading{position:fixed;inset:0;z-index:99;background:#0f1117;display:flex;align-items:center;justify-content:center;
    font-family:'JetBrains Mono',monospace;color:#6e7286;transition:opacity .4s;}
  #loading.hidden{opacity:0;pointer-events:none;}
</style>
</head>
<body>
<div id="loading">Loading map...</div>
<div id="map"></div>
<div id="toolbar">
  <button class="tool-btn active" id="btn-3d">3-D Hills</button>
  <button class="tool-btn" id="btn-exag">Height</button>
  <button class="tool-btn active" id="btn-cells">RMS_B grid</button>
  <button class="tool-btn active" id="btn-path">Smoothed path</button>
  <button class="tool-btn" id="btn-raw">Raw GPS</button>
  <button class="tool-btn active" id="btn-pens">Penetrometer</button>
</div>
<div id="exag-panel">
  <label>Hill height exaggeration <span id="exag-val">__EXAG__x</span></label>
  <input type="range" id="exag-slider" min="1" max="20" value="__EXAG__" step="0.5">
</div>
<div class="panel" id="info">
  <h4>Processing</h4>
  Rows: <b>__N_ROWS__</b><br>
  Tilling rows (speed &gt; __SPEED_MIN__): <b>__N_WORK__</b><br>
  GPS smoothing: <b>RTS Kalman</b>, &sigma;<sub>GPS</sub> = <b>__SIGMA__ m</b><br>
  Grid: <b>__CELL__ m</b> cells, median RMS_ACC_B, &ge; __MINROWS__ rows<br>
  Hardpan: <b>&ge; __HP_PSI__ psi</b> (__HP_KPA__ kPa)<br>
  Penetrometer: <b>__N_PEN__</b> insertions, <b>__N_HP__</b> with hardpan<br>
  <span style="color:#6e7286">Click a cell or a penetrometer point.</span>
</div>
<div class="panel" id="pen-legend">
  <h4>Penetrometer</h4>
  <span class="dot" style="background:#f85149"></span>hardpan (label = top depth)<br>
  <span class="dot" style="background:#3fb950"></span>no hardpan<br>
  <span class="dot" style="background:#8b8fa3"></span>no data
</div>
<div class="panel" id="legend">
  <h4>RMS_ACC_B - cell median</h4>
  <div class="leg-steps">__LEG_STEPS__</div>
  <div class="leg-labels">__LEG_LABELS__</div>
</div>
<script src="https://unpkg.com/maplibre-gl@4.5.0/dist/maplibre-gl.js"></script>
<script>
const CELLS    = __CELLS__;
const PENS     = __PENS__;
const PROFILES = __PROFILES__;
const PATH     = __PATH__;
const RAW      = __RAW__;
const HP_PSI   = __HP_PSI__;
const SHANK_IN = __SHANK__;

function profileSVG(prof, hpTop, hpBot){
  const W=260,H=230,ml=40,mr=12,mt=22,mb=30;
  const pts=prof.d.map((d,i)=>[d,prof.p[i]]).filter(q=>q[1]!==null);
  if(!pts.length) return '<div style="color:#6e7286">no readings</div>';
  const dMax=Math.max(...pts.map(q=>q[0]));
  let pMax=Math.max(HP_PSI*1.3,...pts.map(q=>q[1])); pMax=Math.ceil(pMax/100)*100;
  const X=v=>ml+v/pMax*(W-ml-mr), Y=d=>mt+d/dMax*(H-mt-mb);
  let s=`<svg width="${W}" height="${H}" style="display:block;margin-top:6px">`;
  if(hpTop!==null){
    const y0=Y(hpTop), y1=Y(hpBot===null?hpTop:hpBot);
    s+=`<rect x="${ml}" y="${y0}" width="${W-ml-mr}" height="${Math.max(3,y1-y0)}" fill="rgba(248,81,73,0.18)"/>`;
  }
  const pStep=pMax>600?200:100, dStep=dMax>20?5:2;
  for(let v=0;v<=pMax;v+=pStep){
    s+=`<line x1="${X(v)}" y1="${mt}" x2="${X(v)}" y2="${H-mb}" stroke="rgba(255,255,255,0.07)"/>`;
    s+=`<text x="${X(v)}" y="${mt-6}" fill="#6e7286" font-size="9" text-anchor="middle">${v}</text>`;
  }
  for(let d=0;d<=dMax;d+=dStep){
    s+=`<line x1="${ml}" y1="${Y(d)}" x2="${W-mr}" y2="${Y(d)}" stroke="rgba(255,255,255,0.07)"/>`;
    s+=`<text x="${ml-5}" y="${Y(d)+3}" fill="#6e7286" font-size="9" text-anchor="end">${d}</text>`;
  }
  s+=`<line x1="${X(HP_PSI)}" y1="${mt}" x2="${X(HP_PSI)}" y2="${H-mb}" stroke="#f85149" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  if(SHANK_IN!==null && SHANK_IN<=dMax)
    s+=`<line x1="${ml}" y1="${Y(SHANK_IN)}" x2="${W-mr}" y2="${Y(SHANK_IN)}" stroke="#d29922" stroke-width="1.2" stroke-dasharray="2 3"/>`;
  s+=`<polyline fill="none" stroke="#58a6ff" stroke-width="2" points="${pts.map(q=>X(q[1])+','+Y(q[0])).join(' ')}"/>`;
  pts.forEach(q=>{ s+=`<circle cx="${X(q[1])}" cy="${Y(q[0])}" r="2.2" fill="${q[1]>=HP_PSI?'#f85149':'#58a6ff'}"/>`; });
  s+=`<text x="${(ml+W-mr)/2}" y="${H-8}" fill="#8b8fa3" font-size="10" text-anchor="middle">Cone index (psi)</text>`;
  s+=`<text x="11" y="${(mt+H-mb)/2}" fill="#8b8fa3" font-size="10" text-anchor="middle" transform="rotate(-90 11 ${(mt+H-mb)/2})">Depth (in)</text>`;
  return s+'</svg>';
}

const map=new maplibregl.Map({
  container:'map',
  style:{version:8,glyphs:'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
    sources:{'esri-sat':{type:'raster',tiles:['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],tileSize:256,attribution:'Tiles &copy; Esri'}},
    layers:[{id:'esri-sat',type:'raster',source:'esri-sat'}]},
  center:[__C_LON__,__C_LAT__],zoom:16,pitch:52,bearing:-25,antialias:true
});
map.addControl(new maplibregl.NavigationControl(),'bottom-right');

map.on('load',()=>{
  map.addSource('terrain-dem',{type:'raster-dem',tiles:['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],tileSize:256,maxzoom:15,encoding:'terrarium'});
  map.setTerrain({source:'terrain-dem',exaggeration:__EXAG__});

  map.addSource('cells',{type:'geojson',data:CELLS});
  map.addLayer({id:'cells',type:'fill',source:'cells',paint:{'fill-color':['get','color'],'fill-opacity':0.75}});

  map.addSource('path',{type:'geojson',data:{type:'Feature',geometry:{type:'MultiLineString',coordinates:PATH},properties:{}}});
  map.addLayer({id:'path',type:'line',source:'path',paint:{'line-color':'rgba(255,255,255,0.55)','line-width':1.2}});

  map.addSource('raw',{type:'geojson',data:{type:'Feature',geometry:{type:'MultiPoint',coordinates:RAW},properties:{}}});
  map.addLayer({id:'raw',type:'circle',source:'raw',layout:{visibility:'none'},
    paint:{'circle-radius':1.6,'circle-color':'#ff9100','circle-opacity':0.8}});

  map.addSource('pens',{type:'geojson',data:PENS});
  map.addLayer({id:'pens',type:'circle',source:'pens',paint:{
    'circle-radius':8,'circle-stroke-color':'#0f1117','circle-stroke-width':2.5,
    'circle-color':['match',['get','status'],'hardpan','#f85149','none','#3fb950','#8b8fa3']}});
  map.addLayer({id:'pen-labels',type:'symbol',source:'pens',
    layout:{'text-field':['get','lbl'],'text-font':['Open Sans Semibold'],'text-size':13,'text-offset':[0,-1.5],
            'text-allow-overlap':true,'text-ignore-placement':true},
    paint:{'text-color':'#ffffff','text-halo-color':'#0f1117','text-halo-width':2}});
  document.getElementById('loading').classList.add('hidden');
});

map.on('click','cells',e=>{
  if(map.queryRenderedFeatures(e.point,{layers:['pens']}).length) return;
  const p=e.features[0].properties;
  new maplibregl.Popup({maxWidth:'260px'}).setLngLat(e.lngLat).setHTML(`
    <div class="pt">Grid cell</div>
    RMS_ACC_B median: <span class="pv">${(+p.med).toFixed(3)}</span><br>
    IQR: <span class="pv">${(+p.q25).toFixed(3)} - ${(+p.q75).toFixed(3)}</span><br>
    rows: <span class="pv">${p.n}</span><br>
    median speed: <span class="pv">${(+p.spd).toFixed(2)}</span>`).addTo(map);
});
map.on('click','pens',e=>{
  const p=e.features[0].properties;
  const hpTop=(p.hp_top===undefined||p.hp_top==='null')?null:+p.hp_top;
  const hpBot=(p.hp_bot===undefined||p.hp_bot==='null')?null:+p.hp_bot;
  const hpEnd=(p.hp_end===true||p.hp_end==='true');
  let hp = hpTop===null ? `<span class="pg">no hardpan (&lt; ${HP_PSI} psi)</span>`
         : `<span class="ph">hardpan ${hpTop}" - ${hpBot}"${hpEnd?' (to end of reading)':''}</span>`;
  const rms = (p.rms===undefined||p.rms==='null') ? 'no tilling rows nearby'
            : `${(+p.rms).toFixed(3)} <span style="color:#6e7286">(${p.rms_n} rows)</span>`;
  const shank = (p.psi_shank!==undefined && p.psi_shank!=='null') ? `<br>psi at shank depth: <span class="pv">${p.psi_shank}</span>` : '';
  new maplibregl.Popup({maxWidth:'300px'}).setLngLat(e.features[0].geometry.coordinates).setHTML(`
    <div class="pt">Penetrometer ${p.id}</div>${hp}<br>
    max: <span class="pv">${p.max_psi} psi @ ${p.d_max}"</span>${shank}<br>
    RMS_ACC_B nearby: <span class="pv">${rms}</span>
    ${profileSVG(PROFILES[p.id], hpTop, hpBot)}`).addTo(map);
});
['cells','pens'].forEach(l=>{
  map.on('mouseenter',l,()=>map.getCanvas().style.cursor='pointer');
  map.on('mouseleave',l,()=>map.getCanvas().style.cursor='');
});

function toggle(btnId, layers, startOn, extra){
  let on=startOn; const btn=document.getElementById(btnId);
  btn.addEventListener('click',()=>{
    on=!on; layers.forEach(l=>{ if(map.getLayer(l)) map.setLayoutProperty(l,'visibility',on?'visible':'none'); });
    btn.classList.toggle('active',on); if(extra) extra(on);
  });
}
toggle('btn-cells',['cells'],true,on=>document.getElementById('legend').style.display=on?'block':'none');
toggle('btn-path',['path'],true);
toggle('btn-raw',['raw'],false);
toggle('btn-pens',['pens','pen-labels'],true,on=>document.getElementById('pen-legend').style.display=on?'block':'none');

let is3D=true; const slider=document.getElementById('exag-slider');
document.getElementById('btn-3d').addEventListener('click',()=>{
  is3D=!is3D;
  if(is3D){map.setTerrain({source:'terrain-dem',exaggeration:+slider.value});map.easeTo({pitch:52,bearing:-25,duration:800});}
  else{map.setTerrain(null);map.easeTo({pitch:0,bearing:0,duration:800});}
  document.getElementById('btn-3d').classList.toggle('active',is3D);
});
document.getElementById('btn-exag').addEventListener('click',()=>{
  const p=document.getElementById('exag-panel'); p.style.display=p.style.display==='block'?'none':'block';
});
slider.addEventListener('input',()=>{
  document.getElementById('exag-val').textContent=slider.value+'x';
  if(is3D) map.setTerrain({source:'terrain-dem',exaggeration:+slider.value});
});
</script>
</body>
</html>
"""

dump = lambda o: json.dumps(o, separators=(",", ":"))
rep = {
    "__CELLS__": dump({"type": "FeatureCollection", "features": cell_feats}),
    "__PENS__": dump({"type": "FeatureCollection", "features": pen_feats}),
    "__PROFILES__": dump(pen_profiles),
    "__PATH__": dump(path_lines),
    "__RAW__": dump(raw_pts),
    "__HP_PSI__": f"{HARDPAN_PSI:g}",
    "__HP_KPA__": f"{HARDPAN_PSI * KPA_PER_PSI:.0f}",
    "__SHANK__": "null" if SHANK_DEPTH_IN is None else f"{SHANK_DEPTH_IN:g}",
    "__EXAG__": f"{TERRAIN_EXAGGERATION:g}",
    "__C_LAT__": f"{c_lat:.7f}", "__C_LON__": f"{c_lon:.7f}",
    "__N_ROWS__": f"{n:,}", "__N_WORK__": f"{int(work.sum()):,}",
    "__SPEED_MIN__": f"{SPEED_MIN:g}", "__SIGMA__": f"{sigma_gps:.2f}",
    "__CELL__": f"{CELL_M:g}", "__MINROWS__": f"{MIN_ROWS_PER_CELL}",
    "__N_PEN__": f"{len(val)}", "__N_HP__": f"{n_hp}",
    "__LEG_STEPS__": legend_steps, "__LEG_LABELS__": legend_labels,
}
html = HTML
for k, v in rep.items():
    html = html.replace(k, v)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n{'=' * 60}\n  DONE")
print(f"  Map:         {OUTPUT_HTML}  ({os.path.getsize(OUTPUT_HTML) / 1e6:.1f} MB)")
print(f"  Validation:  {VALIDATION_CSV}")
print(f"  Grid cells:  {CELLS_CSV}")
print(f"{'=' * 60}")