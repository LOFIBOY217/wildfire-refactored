"""
Date Utilities
==============
Date parsing, extraction, and range generation functions.

Consolidates duplicated date functions from:
- pytorch_transformer_fwi20260129.py
- train_s2s_transformer.py
- simple_logistic_7day.py
- evaluate_with_confusion_matrix.py
- verify_data_alignment.py
- download_era5_observations.py
"""

import os
import re
import argparse
from datetime import datetime, timedelta, date
from typing import List, Optional, Union


def parse_date_from_filename(path: str) -> Optional[datetime]:
    """
    Extract datetime from filename containing YYYYMMDD pattern.

    Args:
        path: File path (uses basename for matching)

    Returns:
        datetime object or None if no date found
    """
    m = re.search(r'(20\d{6})', os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")


def extract_date_from_filename(filename: str) -> Optional[date]:
    """
    Extract date object from filename containing 8-digit date pattern.

    Args:
        filename: Filename string (not full path)

    Returns:
        date object or None if no valid date found
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d').date()
        except ValueError:
            return None
    return None


def parse_date_arg(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse command-line date argument supporting YYYYMMDD or YYYY-MM-DD formats.

    Args:
        date_str: Date string or None

    Returns:
        datetime object or None

    Raises:
        argparse.ArgumentTypeError: If date format is invalid
    """
    if date_str is None:
        return None
    date_str = date_str.replace('-', '')
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}, use YYYYMMDD or YYYY-MM-DD"
        )


def generate_date_range(start: Union[datetime, date],
                        end: Union[datetime, date]) -> List[datetime]:
    """
    Generate a list of dates from start to end (inclusive).

    Args:
        start: Start date (datetime or date)
        end: End date (datetime or date)

    Returns:
        List of datetime objects for each day in range
    """
    if isinstance(start, date) and not isinstance(start, datetime):
        start = datetime.combine(start, datetime.min.time())
    if isinstance(end, date) and not isinstance(end, datetime):
        end = datetime.combine(end, datetime.min.time())

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates
