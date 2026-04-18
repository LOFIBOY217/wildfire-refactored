#!/bin/bash
#SBATCH --job-name=wf-swvl2-2000
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/jiaqi217/logs/proc_swvl2_2000_%j.log
#SBATCH --error=/scratch/jiaqi217/logs/proc_swvl2_2000_%j.err
#SBATCH --account=def-inghaw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jiaaqii.huang@mail.utoronto.ca

# ----------------------------------------------------------------
# Process swvl2 (deep soil moisture, layer 2) 2000-2017 GRIBs
# → daily TIF on FWI grid (EPSG:3978)
#
# Input:  data/era5_deep_soil/era5_swvl2_YYYY_MM_DD.grib (2000-2017)
# Output: data/era5_deep_soil/swvl2_YYYYMMDD.tif (FWI grid)
#
# nodata=nan (project standard, matches existing 2018+ swvl2 TIFs).
# After completion, swvl2 has full 2000-2025 EPSG:3978 coverage and
# is ready for 13ch/16ch extended experiments.
# ----------------------------------------------------------------

if [ -z "${SCRATCH:-}" ]; then
    export SCRATCH=/scratch/jiaqi217
fi

[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1 eccodes/2.31.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/eccodes/2.31.0/lib64:${LD_LIBRARY_PATH:-}
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored

echo "============================================="
echo "  Process swvl2 2000-2017 GRIBs → FWI grid"
echo "  Node: $(hostname)  Time: $(date)"
echo "============================================="

python3 -u -c "
import glob, os, re
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

grib_dir = 'data/era5_deep_soil'
gribs = []
for year in range(2000, 2018):
    gribs.extend(sorted(glob.glob(os.path.join(grib_dir, f'era5_swvl2_{year}_*.grib'))))

print(f'Found {len(gribs)} swvl2 GRIB files (2000-2017)')

done, skip, fail = 0, 0, 0
for gi, gf in enumerate(gribs):
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})\.grib$', gf)
    if not m:
        fail += 1
        continue
    date_str = m.group(1) + m.group(2) + m.group(3)
    out_path = os.path.join(grib_dir, f'swvl2_{date_str}.tif')

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

        non_spatial = [d for d in data.dims if d not in ('latitude', 'longitude')]
        if non_spatial:
            arr = data.mean(dim=non_spatial).values.astype(np.float32)
        else:
            arr = data.values.astype(np.float32)
        while arr.ndim > 2:
            arr = arr.mean(axis=0)

        lats = data.latitude.values
        lons = data.longitude.values
        if lats[0] > lats[-1]:
            arr = np.flipud(arr)

        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        h, w = arr.shape
        src_tf = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)

        # IMPORTANT: no src_nodata/dst_nodata — see process_era5_2009_2017_narval.sh
        # for explanation. ERA5 has no real nans; passing them makes reproject
        # refuse to extrapolate at Canada Lambert edges, yielding 24% nan output.
        dst_data = np.full((FWI_H, FWI_W), np.nan, dtype=np.float32)
        reproject(arr, dst_data,
                  src_transform=src_tf, src_crs='EPSG:4326',
                  dst_transform=dst_tf, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(dst_data, 1)
        done += 1

    except Exception as e:
        print(f'  [{gi}] {os.path.basename(gf)}: {e}')
        fail += 1

    if (gi + 1) % 500 == 0:
        print(f'  [{gi+1}/{len(gribs)}] done={done} skip={skip} fail={fail}', flush=True)

print(f'')
print(f'COMPLETE: done={done} skip={skip} fail={fail} total={len(gribs)}')
n_total = len(glob.glob('data/era5_deep_soil/swvl2_*.tif'))
print(f'Total swvl2 TIFs on disk after run: {n_total} (was 2860 before)')
"

PY_EXIT=$?
echo ""
echo "Done: $(date) exit=$PY_EXIT"
exit $PY_EXIT
