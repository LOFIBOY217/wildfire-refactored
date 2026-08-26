# Data sources — where to look when a downloader breaks

Every script in `src/data_ops/download/` pulls from a live external service.
Those services **drift**: layers get renamed, ASCII formats change, files move,
whole APIs get decommissioned. When a downloader that used to work suddenly
returns 400/404, empty data, or garbage, the fix is almost always to find the
new upstream location or request format — not to change our parsing logic.

This file is the registry of **where the authoritative, human-browsable source
lives** for each downloader, so you can go look. Keep it updated: when you fix a
downloader, bump its **Last verified** date and correct its URLs here.

## How to debug a broken downloader

1. **Find the row below** for the script. Open its *Source of truth* URL in a
   browser — that page is what the maintainer sees, and it reflects the current
   dataset, request form, or file listing.
2. **For form/API datasets (Copernicus CDS, ECMWF ECDS):** open the dataset's
   **Download** tab, select the fields, and press **"Show API request code"**.
   That snippet is the ground truth for the request dict — diff it against the
   `request`/`PARAM_SETS` in our script. Field *values* (e.g. variable names,
   `leadtime_hour` format) are the things that silently change.
3. **For GeoServer/WFS/WCS (CWFIS):** hit `GetCapabilities` to confirm the layer
   still exists and check its exact name/paging rules:
   `…/geoserver/ows?service=WFS&version=2.0.0&request=GetCapabilities`
4. **For directory-listing downloads (NBAC, CDEM, WorldPop):** open the parent
   directory URL in a browser and check the current file naming (release stamps
   and year suffixes change on every re-release).
5. **For flat ASCII files (NOAA indices):** `curl` the URL and eyeball the first
   few lines — column layout and missing-value sentinels change without notice.
6. **Check the decommission/announcement page** (ECMWF/Copernicus) if a whole
   endpoint 403s or redirects.

Provenance convention follows the general reproducible-research guidance
(document the origin, the exact endpoint/request, and *when it was last
verified*) and the `pooch` data-registry pattern (name → URL, kept in one place).

---

## Registry

Legend — **Cred:** none = public HTTP; **Last verified:** month we last confirmed
a real download against the live endpoint.

### Canadian Wildland Fire Information System (CWFIS) — `cwfis.cfs.nrcan.gc.ca`

| Script | Dataset | Source of truth | Cred | Last verified |
|---|---|---|---|---|
| `download_hotspots.py` | Satellite hotspots (WFS `public:hotspots`) | [Datamart](https://cwfis.cfs.nrcan.gc.ca/datamart) · [WFS GetCapabilities](https://cwfis.cfs.nrcan.gc.ca/geoserver/ows?service=WFS&version=2.0.0&request=GetCapabilities) | none | 2026-08 |
| `download_ciffc.py` | Fire points (CWFIS WFS + NASA MODIS/FIRMS fallback) | same WFS as above; [NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/) | none / MAP_KEY | 2026-08 |
| `download_fwi_grids.py` | FWI daily rasters (WCS, ~2023→now) | [WCS GetCapabilities](https://cwfis.cfs.nrcan.gc.ca/geoserver/ows?service=WCS&version=1.0.0&request=GetCapabilities) · [live dir](https://cwfis.cfs.nrcan.gc.ca/downloads/cffdrs/) | none | 2026-08 |
| `download_wildfire_data.py` | NFDB points, FBP fuels, NBAC MRB, FWI live/archive | [downloads/](https://cwfis.cfs.nrcan.gc.ca/downloads/) (nfdb/ fuels/ nbac/ cffdrs/) | none | 2026-08 |
| `download_nbac_burn_scars.py` | NBAC annual burn polygons | [downloads/nbac/](https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/) (scrape listing; release stamp changes) | none | 2026-08 |

> **Note:** the `hotspots` layer has no primary key, so any WFS request that
> paginates (`startIndex`) **must** include `sortBy=rep_date` or GeoServer 400s.
> Historical `opendata.nfis.org` NBAC URLs are dead — CWFIS Datamart is canonical.

### NOAA climate indices — `cpc.ncep.noaa.gov`, `ncei.noaa.gov`

| Script | Dataset | Source of truth | Cred | Last verified |
|---|---|---|---|---|
| `download_climate_indices.py` | ONI, PDO, NAO, AO monthly | [CPC teleconnections index](https://www.cpc.ncep.noaa.gov/data/teledoc/telecontents.shtml) · [ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) · [PDO](https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/) | none | 2026-08 |

> **Note:** these are flat ASCII tables and NOAA reformats them without notice
> (ONI is season-coded `SEAS YR …`, NAO/AO are year×month matrices with no
> `Year` header). Missing values are sentinels (`99.99`), not blanks.

### Copernicus Climate / Early-Warning Data Store — `cds.climate.copernicus.eu`, `ewds.climate.copernicus.eu`

| Script | Dataset (collection id) | Source of truth | Cred | Last verified |
|---|---|---|---|---|
| `download_ecmwf_reanalysis_observations.py` | ERA5 single-levels | [CDS dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) · [how-to-api](https://cds.climate.copernicus.eu/how-to-api) | CDS_API_KEY | 2026-08 |
| `download_era5_precip.py` / `download_era5_deep_soil.py` / `download_era5_monthly_batch.py` | ERA5 single-levels (tp / swvl2 / batch) | same as above | CDS_API_KEY | 2026-08 |
| `download_fwi_historical.py` | CEMS fire historical (`cems-fire-historical-v1`) | [EWDS dataset](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1) | CDS_API_KEY | 2026-08 |
| `download_ecmwf_s2s_fire.py` | CEMS fire seasonal | [EWDS dataset](https://ewds.climate.copernicus.eu/datasets/cems-fire-seasonal) | CDS_API_KEY | 2026-08 |

> **Note:** CEMS fire moved CDS→EWDS and EWDS **retired the `netcdf_legacy`
> format** (use `netcdf`). All of these use the `cdsapi` client; the token is
> the unified ECMWF personal access token.

### ECMWF Data Store (ECDS) — `ecds.ecmwf.int`

| Script | Dataset (collection id) | Source of truth | Cred | Last verified |
|---|---|---|---|---|
| `download_ecmwf_s2s.py` | S2S real-time (`s2s-forecasts`) | [dataset Download tab](https://ecds.ecmwf.int/datasets/s2s-forecasts?tab=download) · [how-to-api](https://ecds.ecmwf.int/how-to-api) | ECDS/CDS token | 2026-08 |
| `download_ecmwf_hres_7day.py` | TIGGE (`tigge-forecasts`) | [dataset Download tab](https://ecds.ecmwf.int/datasets/tigge-forecasts?tab=download) | ECDS/CDS token | 2026-08 |

> **Note:** the legacy ECMWF Web-API (`api.ecmwf.int/v1`, `ecmwfapi` library)
> was **decommissioned 2026-05-27**; S2S/TIGGE now live on ECDS and are pulled
> with `cdsapi`. Each dataset needs its **licence accepted once** on the
> Download tab. Deprecations are announced on the
> [Decommissioning page](https://confluence.ecmwf.int/display/DAC/Decommissioning+of+ECMWF+Public+Datasets+Service).

### NASA FIRMS / Earthdata

| Script | Dataset | Source of truth | Cred | Last verified |
|---|---|---|---|---|
| `download_firms_viirs.py` | VIIRS S-NPP active fire (2012–2017) | [FIRMS API docs](https://firms.modaps.eosdis.nasa.gov/api/area/) · [get MAP_KEY](https://firms.modaps.eosdis.nasa.gov/api/map_key/) | FIRMS MAP_KEY | 2026-08 |
| `download_modis_ndvi.py` | MOD13A2 v6.1 NDVI/EVI | [LP DAAC MOD13A2](https://lpdaac.usgs.gov/products/mod13a2v061/) · [Earthdata Search](https://search.earthdata.nasa.gov/) | Earthdata (`~/.netrc`) | 2026-08 |
| `download_srtm_slope.py` | SRTMGL1 v003 elevation → slope | [LP DAAC SRTMGL1](https://lpdaac.usgs.gov/products/srtmgl1v003/) | Earthdata (`~/.netrc`) | 2026-08 |

> **Note:** MODIS/SRTM use the `earthaccess` library, which reads the
> `urs.earthdata.nasa.gov` login from `~/.netrc`.

### Other public providers

| Script | Dataset | Source of truth | Cred | Last verified |
|---|---|---|---|---|
| `download_population.py` | WorldPop 2020 Canada 1 km density | [WorldPop hub](https://hub.worldpop.org/geodata/summary?id=24777) · [GIS dir](https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km/2020/CAN/) | none | 2026-08 |
| `download_cdem.py` | NRCan CDEM elevation tiles | [NRCan elevation FTP](https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/) | none | 2026-08 |
| `download_goes_glm.py` | GOES-16/18 GLM lightning | [AWS Open Data registry](https://registry.opendata.aws/noaa-goes/) (S3 `noaa-goes16`/`noaa-goes18`) | none (anon S3) | 2026-08 |

> **Note:** `download_goes_glm.py` needs `netCDF4` installed, or it silently
> writes empty rasters — it now pre-flights the import.

---

*When you touch a downloader: update the row's **Last verified** date and any
changed URL here, and mirror the one-line pointer in the script's own docstring.*
