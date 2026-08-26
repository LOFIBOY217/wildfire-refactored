#!/usr/bin/env python3
"""
Shared helpers for the data-download scripts in this package.

Kept intentionally small: only genuinely identical boilerplate lives here (the
Copernicus CDS client factory and a daily date-range generator). Each
downloader keeps its own dataset-specific request and parsing logic.
"""

import os
from datetime import datetime, timedelta


def make_cds_client(api_key=None, **client_kwargs):
    """Create a Copernicus CDS API client.

    Args:
        api_key: CDS API key. If falsy, read from the CDS_API_KEY environment
            variable; an empty value lets cdsapi fall back to ~/.cdsapirc.
            Never hardcode a key.
        **client_kwargs: forwarded to cdsapi.Client (e.g. quiet=False).
    """
    import cdsapi
    if not api_key:
        api_key = os.environ.get("CDS_API_KEY", "")
    return cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=api_key,
        **client_kwargs,
    )


# ECMWF Data Store (ECDS) API root. ECDS hosts the S2S and TIGGE datasets,
# which were migrated off the retired ECMWF Web-API (api.ecmwf.int/v1) in 2026.
ECDS_API_URL = "https://ecds.ecmwf.int/api"


def make_ecds_client(api_key=None, **client_kwargs):
    """Create a client for the ECMWF Data Store (ECDS).

    ECDS speaks the same protocol as the Copernicus CDS, so we reuse the
    ``cdsapi`` client pointed at the ECDS API root. The access token is the
    unified ECMWF personal access token (the same one used for CDS/ADS/CEMS);
    ECDS just needs the URL swapped.

    Args:
        api_key: ECDS personal access token. If falsy, read from the
            ECDS_API_KEY then CDS_API_KEY environment variables; if still empty,
            ``cdsapi`` falls back to the ``key:`` line of ~/.cdsapirc (the token
            is store-agnostic, so a CDS-configured rc file works). Never
            hardcode a key.
        **client_kwargs: forwarded to cdsapi.Client (e.g. quiet=False).
    """
    import cdsapi
    if not api_key:
        api_key = os.environ.get("ECDS_API_KEY") or os.environ.get("CDS_API_KEY", "")
    if api_key:
        return cdsapi.Client(url=ECDS_API_URL, key=api_key, **client_kwargs)
    # No explicit key: let cdsapi read the token from ~/.cdsapirc, but force the
    # ECDS API root.
    return cdsapi.Client(url=ECDS_API_URL, **client_kwargs)


def generate_date_list(start_date, end_date):
    """Generate list of YYYY-MM-DD strings between start and end (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates
