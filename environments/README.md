# Environment Profiles

This repository uses one base conda environment plus profile-specific overlays.

## Base (common for all machines)

```bash
conda env create -f environment.yml
```

## Local profile (with pygrib)

Use this on machines where `pygrib` is available:

```bash
conda env update -n wildfire -f environments/environment.local-pygrib.yml
```

## HPC profile (without pygrib, using cfgrib)

Use this on HPC where `pygrib` cannot be installed:

```bash
conda env update -n wildfire -f environments/environment.hpc-cfgrib.yml
```

## Verify installed GRIB backend

```bash
python - <<'PY'
import importlib
print('pygrib:', importlib.util.find_spec('pygrib') is not None)
print('cfgrib:', importlib.util.find_spec('cfgrib') is not None)
PY
```
