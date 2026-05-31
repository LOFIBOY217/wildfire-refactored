#!/bin/bash
#SBATCH --job-name=wf-label-slices
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=/scratch/jiaqi217/logs/label_slices_%j.log
#SBATCH --account=def-inghaw
set -uo pipefail
export SCRATCH=${SCRATCH:-/scratch/jiaqi217}
[[ -z "$(command -v module)" ]] && source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1
source $SCRATCH/venv-wildfire/bin/activate
cd $SCRATCH/wildfire-refactored
python3 << 'PY'
import numpy as np, rasterio
from datetime import date
LBL='data/fire_labels/fire_labels_nbac_nfdb_2000-05-01_2025-12-21_2281x2709_r14.npy'
REF='outputs/v3_9ch_enc21_12y_2014_fire_prob/20230515/fire_prob_lead14d_20230529.tif'
S0=date(2000,5,1)
labels=np.load(LBL,mmap_mode='r')
with rasterio.open(REF) as r: prof=r.profile.copy()
prof.update(dtype='uint8',count=1,compress='deflate',nodata=0)
for tag,t0,t1 in [('20220829_20220929',date(2022,8,29),date(2022,9,29)),
                  ('20230829_20230929',date(2023,8,29),date(2023,9,29))]:
    i0=(t0-S0).days; i1=(t1-S0).days
    u=(labels[i0:i1+1].sum(0)>0).astype('uint8')
    out=f'outputs/fire_actual_{tag}.tif'
    with rasterio.open(out,'w',**prof) as d: d.write(u,1)
    print(tag,'fire px',int(u.sum()),'->',out)
PY
echo "done"
