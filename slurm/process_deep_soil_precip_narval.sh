#!/bin/bash
#SBATCH --job-name=wf-proc-soil-tp
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/proc_soil_tp_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/proc_soil_tp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Process downloaded deep_soil (swvl2) and precip (tp) GRIBs
# → daily TIF on FWI grid (EPSG:3978, 2709x2281)
#
# Input:  data/era5_deep_soil/era5_swvl2_*.grib (single-var, single-day)
#         data/era5_precip/era5_tp_*.grib        (single-var, single-day)
# Output: data/era5_deep_soil/swvl2_YYYYMMDD.tif (FWI grid)
#         data/era5_precip/tp_YYYYMMDD.tif       (FWI grid)
# ----------------------------------------------------------------

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:${LD_LIBRARY_PATH:-}
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  Process deep_soil + precip GRIBs → FWI grid TIFs"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -c "
import glob, os, re, sys
import numpy as np
import cfgrib
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

FWI_CRS = 'EPSG:3978'
FWI_W, FWI_H = 2709, 2281
FWI_BOUNDS = (-2378164.0, -707617.0, 3039835.0, 3854382.0)
dst_tf = from_bounds(*FWI_BOUNDS, FWI_W, FWI_H)
dst_crs = CRS.from_string(FWI_CRS)

profile = dict(driver='GTiff', dtype='float32', width=FWI_W, height=FWI_H,
               count=1, crs=dst_crs, transform=dst_tf, nodata=np.nan, compress='lzw')

def process_var(grib_dir, var_prefix, out_var):
    pattern = os.path.join(grib_dir, f'era5_{var_prefix}_*.grib')
    gribs = sorted(glob.glob(pattern))
    print(f'\n{\"=\" * 60}')
    print(f'  Processing {out_var}: {len(gribs)} GRIB files in {grib_dir}')
    print(f'{\"=\" * 60}')

    done, skip, fail = 0, 0, 0
    for gi, gf in enumerate(gribs):
        # Extract date: era5_swvl2_2024_07_28.grib → 20240728
        m = re.search(r'(\d{4})_(\d{2})_(\d{2})\.grib$', gf)
        if not m:
            fail += 1
            continue
        date_str = m.group(1) + m.group(2) + m.group(3)
        out_path = os.path.join(grib_dir, f'{out_var}_{date_str}.tif')

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            skip += 1
            continue

        try:
            datasets = cfgrib.open_datasets(gf)
            data = None
            for ds in datasets:
                for vname in ds.data_vars:
                    data = ds[vname]
                    break
                if data is not None:
                    break

            if data is None:
                print(f'  [{gi}] {os.path.basename(gf)}: no data vars')
                fail += 1
                continue

            # Daily average: collapse ALL non-spatial dims (time, step, etc.)
            non_spatial = [d for d in data.dims if d not in ('latitude', 'longitude')]
            if non_spatial:
                arr = data.mean(dim=non_spatial).values.astype(np.float32)
            else:
                arr = data.values.astype(np.float32)

            # Safety: squeeze any remaining singular dims
            while arr.ndim > 2:
                arr = arr.mean(axis=0)

            lats = data.latitude.values if 'latitude' in data.coords else data.coords['latitude'].values
            lons = data.longitude.values if 'longitude' in data.coords else data.coords['longitude'].values

            # ERA5 lat descending
            if lats[0] > lats[-1]:
                arr = np.flipud(arr)

            lat_min, lat_max = float(lats.min()), float(lats.max())
            lon_min, lon_max = float(lons.min()), float(lons.max())
            h, w = arr.shape
            src_tf = from_bounds(lon_min, min(lat_min, lat_max),
                                 lon_max, max(lat_min, lat_max), w, h)

            # Reproject to FWI grid
            dst_data = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
            reproject(arr, dst_data,
                      src_transform=src_tf, src_crs='EPSG:4326',
                      dst_transform=dst_tf, dst_crs=dst_crs,
                      resampling=Resampling.bilinear,
                      src_nodata=np.nan, dst_nodata=np.nan)

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(dst_data, 1)
            done += 1

        except Exception as e:
            print(f'  [{gi}] {os.path.basename(gf)}: {e}')
            fail += 1

        if (done + skip) % 200 == 0 and done > 0:
            print(f'  {out_var}: processed {done}, skipped {skip}, failed {fail} / {gi+1}')

    print(f'  {out_var} DONE: processed={done} skipped={skip} failed={fail} total={len(gribs)}')
    return done, skip, fail

# Process both variables
d1, s1, f1 = process_var('data/era5_deep_soil', 'swvl2', 'swvl2')
d2, s2, f2 = process_var('data/era5_precip', 'tp', 'tp')

print(f'\n{\"=\" * 60}')
print(f'  SUMMARY')
print(f'  deep_soil (swvl2): {d1} new + {s1} existing = {d1+s1} TIFs  ({f1} failed)')
print(f'  precip (tp):       {d2} new + {s2} existing = {d2+s2} TIFs  ({f2} failed)')
n_soil = len(glob.glob('data/era5_deep_soil/swvl2_*.tif'))
n_tp = len(glob.glob('data/era5_precip/tp_*.tif'))
print(f'  Final: swvl2={n_soil} TIFs, tp={n_tp} TIFs on FWI grid')
print(f'{\"=\" * 60}')
" 2>&1

echo ""
echo "Done: $(date)"
