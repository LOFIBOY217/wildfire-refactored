#!/usr/bin/env python3
"""
Download FWI data from CWFIS WCS (Web Coverage Service).

Usage:
    python -m src.data_ops.download.fwi_wcs 20240901
    python -m src.data_ops.download.fwi_wcs 20240901 20241231
    python -m src.data_ops.download.fwi_wcs --config configs/custom.yaml 20240901
"""

import argparse
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from src.config import load_config, get_path, add_config_argument


# ------------------------------------------------------------------ #
# FWI Downloader
# ------------------------------------------------------------------ #

class FWIDownloader:
    """Download FWI rasters from CWFIS WCS."""

    def __init__(self, output_dir="fwi_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CWFIS WCS service
        self.wcs_base = (
            "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs"
            "?service=WCS&version=1.0.0&request=GetCoverage"
            "&format=GeoTIFF"
            "&crs=EPSG:3978"
            "&bbox=-2378164,-707617,3039835,3854382"
            "&width=2709&height=2281"
        )

    @staticmethod
    def _to_time_value(date_str):
        """Convert YYYYMMDD -> YYYY-MM-DD for WCS time dimension."""
        return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    def _build_urls(self, date_str):
        """
        Build candidate URLs for one date.

        Preferred (new): coverage=public:fwi&time=YYYY-MM-DD
        Fallback (legacy): coverage=public:fwiYYYYMMDD
        """
        time_value = self._to_time_value(date_str)
        return [
            f"{self.wcs_base}&coverage=public:fwi&time={time_value}",
            f"{self.wcs_base}&coverage=public:fwi{date_str}",
        ]

    def download_date(self, date_str):
        """
        Download FWI for single date.

        Args:
            date_str: Date in YYYYMMDD format.

        Returns:
            True on success, False on failure.
        """
        output_path = self.output_dir / f"fwi_{date_str}.tif"

        if output_path.exists():
            print(f"[SKIP] {date_str}")
            return True

        urls = self._build_urls(date_str)
        print(f"[DOWNLOADING] {date_str}...", end=" ", flush=True)

        for idx, url in enumerate(urls):
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'FWI-Downloader/1.0'},
                )

                with urllib.request.urlopen(req, timeout=30) as response:
                    data = response.read()

                if len(data) < 1000:
                    # Try next URL variant if response looks like an error payload.
                    if idx < len(urls) - 1:
                        continue
                    print("[FAIL] (too small)")
                    return False

                with open(output_path, 'wb') as f:
                    f.write(data)

                size_mb = len(data) / 1024 / 1024
                if idx == 0:
                    print(f"[OK] {size_mb:.1f}MB")
                else:
                    print(f"[OK] {size_mb:.1f}MB (legacy endpoint)")
                return True

            except urllib.error.HTTPError:
                # Try fallback URL if available.
                if idx < len(urls) - 1:
                    continue
                print("[FAIL] (http error)")
                return False
            except Exception as e:
                if idx < len(urls) - 1:
                    continue
                print(f"[FAIL] ({e})")
                return False

        print("[FAIL]")
        return False

    def download_range(self, start_date, end_date):
        """
        Download FWI for a date range.

        Args:
            start_date: Start date in YYYYMMDD format.
            end_date: End date in YYYYMMDD format.
        """
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        current = start
        success = 0
        failed = 0
        skipped = 0

        while current <= end:
            date_str = current.strftime("%Y%m%d")

            if (self.output_dir / f"fwi_{date_str}.tif").exists():
                skipped += 1
            elif self.download_date(date_str):
                success += 1
            else:
                failed += 1

            current += timedelta(days=1)

        # Summary
        total = success + failed + skipped
        print(f"\n{'='*40}")
        print(f"Total: {total} | OK: {success} | FAIL: {failed} | SKIP: {skipped}")
        print(f"{'='*40}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Download FWI data from CWFIS WCS service"
    )
    add_config_argument(parser)

    parser.add_argument(
        "start_date",
        help="Start date in YYYYMMDD format",
    )
    parser.add_argument(
        "end_date", nargs="?", default=None,
        help="End date in YYYYMMDD format (optional; defaults to start_date)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load config and resolve output directory
    cfg = load_config(args.config)
    fwi_dir = get_path(cfg, "fwi_dir")

    downloader = FWIDownloader(output_dir=fwi_dir)

    start_date = args.start_date
    end_date = args.end_date if args.end_date else start_date

    if start_date == end_date:
        downloader.download_date(start_date)
    else:
        downloader.download_range(start_date, end_date)


if __name__ == "__main__":
    main()
