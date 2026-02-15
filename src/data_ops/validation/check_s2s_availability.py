#!/usr/bin/env python3
"""
Check S2S training data availability.

Validates whether each candidate issue date has:
1) encoder FWI history,
2) ECMWF S2S decoder inputs (target-date folders and 5 variables),
3) target FWI labels.

Usage:
    python -m src.data_ops.validation.check_s2s_availability \
        --fwi-start 20250101 --fwi-end 20251231
"""

import argparse
import glob
import os
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

try:
    from src.config import load_config, get_path, add_config_argument
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "config.py").exists():
            sys.path.insert(0, str(parent))
            break
    from src.config import load_config, get_path, add_config_argument

from src.utils.date_utils import parse_date_arg, parse_date_from_filename, generate_date_range


ECMWF_VARS = ["2t", "2d", "tcw", "sm20", "st20"]


def build_fwi_date_set(fwi_dir):
    """Return a set of available FWI dates parsed from .tif filenames."""
    files = glob.glob(os.path.join(fwi_dir, "*.tif"))
    dates = set()
    for filepath in files:
        d = parse_date_from_filename(filepath)
        if d is not None:
            dates.add(d)
    return dates


def validate_ecmwf_issue(ecmwf_dir, issue_date, lead_start, lead_end):
    """
    Validate one issue date's ECMWF decoder inputs.

    Returns:
        (ok: bool, reason: str)
        reason in {"issue_dir", "target_dir", "var_file", "ok"}
    """
    issue_str = issue_date.strftime("%Y%m%d")
    issue_dir = os.path.join(ecmwf_dir, issue_str)
    if not os.path.isdir(issue_dir):
        return False, "issue_dir"

    for target_date in generate_date_range(
        issue_date + timedelta(days=lead_start),
        issue_date + timedelta(days=lead_end),
    ):
        target_str = target_date.strftime("%Y%m%d")
        target_dir = os.path.join(issue_dir, target_str)
        if not os.path.isdir(target_dir):
            return False, "target_dir"

        for var in ECMWF_VARS:
            var_path = os.path.join(target_dir, f"{var}.tif")
            if not os.path.isfile(var_path):
                return False, "var_file"

    return True, "ok"


def check_s2s_availability(
    fwi_dir,
    ecmwf_dir,
    fwi_start,
    fwi_end,
    encoder_days,
    lead_start,
    lead_end,
    max_examples=10,
):
    """Run full S2S data availability check and print summary."""
    if not os.path.isdir(fwi_dir):
        raise FileNotFoundError(f"FWI dir not found: {fwi_dir}")
    if not os.path.isdir(ecmwf_dir):
        raise FileNotFoundError(f"ECMWF dir not found: {ecmwf_dir}")

    print("=" * 72)
    print("S2S DATA AVAILABILITY CHECK")
    print("=" * 72)
    print(f"FWI dir:    {fwi_dir}")
    print(f"ECMWF dir:  {ecmwf_dir}")
    print(f"FWI target: {fwi_start.strftime('%Y-%m-%d')} -> {fwi_end.strftime('%Y-%m-%d')}")
    print(
        "Config: "
        f"encoder_days={encoder_days}, lead_start={lead_start}, lead_end={lead_end}"
    )

    fwi_dates = build_fwi_date_set(fwi_dir)
    print(f"Indexed FWI dates: {len(fwi_dates)}")

    issue_start = fwi_start - timedelta(days=lead_start)
    issue_end = fwi_end - timedelta(days=lead_end)
    issue_dates = generate_date_range(issue_start, issue_end)

    print(
        f"Candidate issue dates: {len(issue_dates)} "
        f"({issue_start.strftime('%Y-%m-%d')} -> {issue_end.strftime('%Y-%m-%d')})"
    )
    print("-" * 72)

    skipped = Counter()
    valid_count = 0
    examples = {
        "encoder": [],
        "target": [],
        "ecmwf_issue_dir": [],
        "ecmwf_target_dir": [],
        "ecmwf_var_file": [],
    }

    for i, issue_date in enumerate(issue_dates, 1):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(issue_dates)}")

        encoder_start = issue_date - timedelta(days=encoder_days)
        encoder_end = issue_date - timedelta(days=1)
        encoder_window = generate_date_range(encoder_start, encoder_end)
        if not all(d in fwi_dates for d in encoder_window):
            skipped["encoder"] += 1
            if len(examples["encoder"]) < max_examples:
                examples["encoder"].append(issue_date.strftime("%Y-%m-%d"))
            continue

        target_start = issue_date + timedelta(days=lead_start)
        target_end = issue_date + timedelta(days=lead_end)
        target_window = generate_date_range(target_start, target_end)
        if not all(d in fwi_dates for d in target_window):
            skipped["target"] += 1
            if len(examples["target"]) < max_examples:
                examples["target"].append(issue_date.strftime("%Y-%m-%d"))
            continue

        ok, reason = validate_ecmwf_issue(ecmwf_dir, issue_date, lead_start, lead_end)
        if not ok:
            skipped[reason] += 1
            if reason == "issue_dir" and len(examples["ecmwf_issue_dir"]) < max_examples:
                examples["ecmwf_issue_dir"].append(issue_date.strftime("%Y-%m-%d"))
            elif reason == "target_dir" and len(examples["ecmwf_target_dir"]) < max_examples:
                examples["ecmwf_target_dir"].append(issue_date.strftime("%Y-%m-%d"))
            elif reason == "var_file" and len(examples["ecmwf_var_file"]) < max_examples:
                examples["ecmwf_var_file"].append(issue_date.strftime("%Y-%m-%d"))
            continue

        valid_count += 1

    total = len(issue_dates)
    pct = 100.0 * valid_count / max(total, 1)

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"Valid samples: {valid_count}/{total} ({pct:.1f}%)")
    print(f"Skipped - missing encoder FWI:      {skipped['encoder']}")
    print(f"Skipped - missing target FWI:       {skipped['target']}")
    print(f"Skipped - missing ECMWF issue dir:  {skipped['issue_dir']}")
    print(f"Skipped - missing ECMWF target dir: {skipped['target_dir']}")
    print(f"Skipped - missing ECMWF var file:   {skipped['var_file']}")

    print("\nExamples of failed issue dates:")
    for key, label in [
        ("encoder", "encoder FWI missing"),
        ("target", "target FWI missing"),
        ("ecmwf_issue_dir", "ECMWF issue dir missing"),
        ("ecmwf_target_dir", "ECMWF target dir missing"),
        ("ecmwf_var_file", "ECMWF var file missing"),
    ]:
        vals = examples[key]
        if vals:
            print(f"- {label}: {', '.join(vals)}")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether S2S training inputs/labels are fully available",
    )
    add_config_argument(parser)
    parser.add_argument("--fwi-dir", type=str, default=None, help="Override fwi_dir path")
    parser.add_argument("--ecmwf-dir", type=str, default=None, help="Override ecmwf_dir path")
    parser.add_argument("--fwi-start", type=str, required=True, help="YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--fwi-end", type=str, required=True, help="YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--encoder-days", type=int, default=7)
    parser.add_argument("--lead-start", type=int, default=14)
    parser.add_argument("--lead-end", type=int, default=46)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Max example failed issue dates to print per category",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fwi_dir = args.fwi_dir or get_path(cfg, "fwi_dir")
    ecmwf_dir = args.ecmwf_dir or get_path(cfg, "ecmwf_dir")
    fwi_start = parse_date_arg(args.fwi_start)
    fwi_end = parse_date_arg(args.fwi_end)

    if fwi_start is None or fwi_end is None:
        raise ValueError("Both --fwi-start and --fwi-end are required")
    if fwi_end < fwi_start:
        raise ValueError("--fwi-end must be >= --fwi-start")
    if args.lead_end < args.lead_start:
        raise ValueError("--lead-end must be >= --lead-start")
    if args.encoder_days <= 0:
        raise ValueError("--encoder-days must be > 0")

    check_s2s_availability(
        fwi_dir=fwi_dir,
        ecmwf_dir=ecmwf_dir,
        fwi_start=fwi_start,
        fwi_end=fwi_end,
        encoder_days=args.encoder_days,
        lead_start=args.lead_start,
        lead_end=args.lead_end,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
