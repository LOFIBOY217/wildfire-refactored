#!/bin/bash
#SBATCH --job-name=wf-nfdb-clim
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/nfdb_clim_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/nfdb_clim_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# A/B TEST: Build fire_clim_annual from NFDB (human-reported)
# instead of hotspot CSV (satellite-detected).
#
# Motivation: hotspot detection shows 13× growth 2000→2023 due to
# satellite technology upgrade (MODIS → VIIRS). NFDB is agency-
# reported fire starts, methodology stable over decades.
#
# Pipeline:
#   1. Parse NFDB_point shapefile → CSV (lat/lon/date 2000-2025)
#   2. Rebuild fire_clim_upto_{Y}.tif in data/fire_clim_annual_nfdb/
#   3. (Later) Train enc28 × 2000-2025 with --fire_clim_dir
#      pointing to new dir → A/B compare vs hotspot version
# ----------------------------------------------------------------

set -uo pipefail

export SCRATCH=${SCRATCH:-/scratch/jiaqi217}

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1

source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
export PYTHONPATH=$SCRATCH/wildfire-refactored:$PYTHONPATH
export PROJ_DATA=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj
export PYTHONUNBUFFERED=1

NFDB_DIR="data/nfdb"
NFDB_CSV="data/hotspot/nfdb_fires_2000_2025.csv"
OUT_DIR="data/fire_clim_annual_nfdb"

echo "============================================="
echo "  NFDB-based fire_clim_annual builder"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

# ─── STEP 1: Extract shapefile from zip ──────────────────────
echo ""
echo "[1/3] Unzipping NFDB_point.zip..."
python3 -u -c "
import zipfile, os, sys
zf = '$NFDB_DIR/NFDB_point.zip'
out = '$NFDB_DIR/extract'
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(zf) as z:
    z.extractall(out)
# List shp file
for f in os.listdir(out):
    if f.endswith('.shp'):
        print(f'  Shapefile: {f}')
        break
" || exit 1

SHP=$(ls $NFDB_DIR/extract/*.shp | head -1)
echo "  Using: $SHP"

# ─── STEP 2: Parse shapefile → CSV ────────────────────────────
echo ""
echo "[2/3] Parse NFDB shapefile → CSV (2000-2025, Canada bbox)..."
mkdir -p data/hotspot

python3 -u -c "
import geopandas as gpd
import pandas as pd
import numpy as np

shp = '$SHP'
print(f'  Reading {shp}...')
gdf = gpd.read_file(shp)
print(f'  Total records: {len(gdf):,}')
print(f'  Columns: {list(gdf.columns)[:20]}')
print(f'  CRS: {gdf.crs}')

# Find lat/lon columns (NFDB uses LATITUDE/LONGITUDE)
lat_col = next((c for c in gdf.columns if c.upper() in ('LATITUDE', 'LAT', 'Y')), None)
lon_col = next((c for c in gdf.columns if c.upper() in ('LONGITUDE', 'LON', 'LONG', 'X')), None)

# Date: try REP_DATE (report date), or build from YEAR/MONTH/DAY
date_col = None
for c in ['REP_DATE', 'IGN_DATE', 'STARTDATE', 'DATE_OUT']:
    if c in gdf.columns:
        date_col = c
        break

if lat_col is None or lon_col is None:
    # Fall back: extract from geometry
    print(f'  No lat/lon columns; using geometry centroid')
    # Reproject to WGS84 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf_wgs = gdf.to_crs('EPSG:4326')
    else:
        gdf_wgs = gdf
    gdf['LATITUDE']  = gdf_wgs.geometry.y
    gdf['LONGITUDE'] = gdf_wgs.geometry.x
    lat_col, lon_col = 'LATITUDE', 'LONGITUDE'

print(f'  lat={lat_col}  lon={lon_col}  date={date_col}')

# Build date column
if date_col and date_col in gdf.columns:
    gdf['acq_date'] = pd.to_datetime(gdf[date_col], errors='coerce')
elif 'YEAR' in gdf.columns and 'MONTH' in gdf.columns and 'DAY' in gdf.columns:
    gdf['acq_date'] = pd.to_datetime(
        gdf[['YEAR', 'MONTH', 'DAY']].rename(columns={'YEAR':'year','MONTH':'month','DAY':'day'}),
        errors='coerce')
else:
    print('  ERROR: cannot determine date column')
    import sys; sys.exit(1)

# Filter: 2000-2025, valid lat/lon, Canada bbox
df = gdf[['acq_date', lat_col, lon_col]].copy()
df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
df = df.dropna(subset=['acq_date', 'latitude', 'longitude'])
df['year'] = df['acq_date'].dt.year
df = df[(df.year >= 2000) & (df.year <= 2025)]
df = df[(df.latitude > 40) & (df.latitude < 85)]
df = df[(df.longitude > -145) & (df.longitude < -50)]

# Save in CWFIS-compatible format (latitude, longitude, acq_date)
df['acq_date'] = df['acq_date'].dt.strftime('%Y-%m-%d')
df = df[['latitude', 'longitude', 'acq_date']]
df.to_csv('$NFDB_CSV', index=False)
print(f'  Saved: $NFDB_CSV')
print(f'  Records: {len(df):,}')

# Per-year count
print()
print('  NFDB fires per year:')
for y in range(2000, 2026):
    yr = df[df.acq_date.str.startswith(str(y))]
    print(f'    {y}: {len(yr):>6,}')
" || exit 1

# ─── STEP 3: Rebuild fire_clim_annual using NFDB CSV ─────────
echo ""
echo "[3/3] Rebuild fire_clim_annual → $OUT_DIR..."

# Temporarily override hotspot_csv path via custom call
python3 -u -c "
import sys
sys.path.insert(0, '.')
from src.data_ops.processing.make_fire_climatology import build_fire_climatology

ref = 'data/fwi_data/fwi_20250615.tif'
months = list(range(5, 11))  # May-Oct

for target_year in range(2000, 2026):
    prior_years = list(range(2000, target_year))
    out_path = '$OUT_DIR/fire_clim_upto_{}.tif'.format(target_year)

    import os
    os.makedirs('$OUT_DIR', exist_ok=True)

    if not prior_years:
        # No prior data → all-zero map
        import rasterio, numpy as np
        with rasterio.open(ref) as src:
            profile = src.profile.copy()
            h, w = src.height, src.width
        profile.update(dtype='float32', count=1, compress='lzw', nodata=None)
        profile.pop('photometric', None)
        out = np.zeros((h, w), dtype=np.float32)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out, 1)
        print(f'  [zero prior] {target_year}')
        continue

    print(f'  [{target_year}] prior years = {prior_years[0]}..{prior_years[-1]}')
    build_fire_climatology(
        hotspot_csv='$NFDB_CSV',
        fwi_reference_tif=ref,
        years=prior_years,
        months=months,
        output_path=out_path,
    )

print()
print('DONE. Output files:')
import glob
for f in sorted(glob.glob('$OUT_DIR/*.tif')):
    print(f'  {f}')
" || exit 1

PY_EXIT=$?
echo ""
echo "============================================="
echo "  Done: $(date) exit=$PY_EXIT"
echo "============================================="
exit $PY_EXIT
