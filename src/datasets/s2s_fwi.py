"""
S2S FWI Dataset
===============
Dataset combining FWI history (encoder) + ECMWF forecasts (decoder) for
subseasonal-to-seasonal wildfire prediction.

Based on train_s2s_transformer.py S2SFWIDataset.
"""

import os
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.raster_io import load_fwi_file, load_rasterio_file, clean_nodata
from src.utils.date_utils import parse_date_from_filename, generate_date_range
from src.utils.patch_utils import patchify

# ECMWF variable names
ECMWF_VARS = ['2t', '2d', 'tcw', 'sm20', 'st20']


def build_fwi_index(fwi_dir):
    """
    Build {datetime: filepath} index for FWI files.

    Returns:
        dict mapping datetime -> file path
    """
    files = glob.glob(os.path.join(fwi_dir, "*.tif"))
    index = {}
    for filepath in files:
        date = parse_date_from_filename(filepath)
        if date is not None:
            index[date] = filepath

    print(f"[FWI Index] Found {len(index)} FWI files")
    if index:
        dates = sorted(index.keys())
        print(f"[FWI Index] Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    return index


def load_fwi_sequence(fwi_index, start_date, end_date):
    """
    Load a contiguous FWI sequence.

    Returns:
        (num_days, H, W) array, or None if any date is missing
    """
    dates = generate_date_range(start_date, end_date)
    frames = []
    for date in dates:
        if date not in fwi_index:
            return None
        arr = load_fwi_file(fwi_index[date])
        frames.append(arr)
    return np.stack(frames, axis=0)


def load_ecmwf_sequence(ecmwf_dir, issue_date, lead_start=14, lead_end=46):
    """
    Load ECMWF forecast sequence for a given issue date.

    Args:
        ecmwf_dir: Root ECMWF directory
        issue_date: Forecast issue date
        lead_start: Start lead time (days)
        lead_end: End lead time (days)

    Returns:
        (num_days, H, W, 5) array, or None if data incomplete
    """
    issue_str = issue_date.strftime('%Y%m%d')
    issue_dir = os.path.join(ecmwf_dir, issue_str)

    if not os.path.exists(issue_dir):
        return None

    target_dates = [issue_date + timedelta(days=i) for i in range(lead_start, lead_end + 1)]

    frames = []
    for target_date in target_dates:
        target_str = target_date.strftime('%Y%m%d')
        target_dir = os.path.join(issue_dir, target_str)

        if not os.path.exists(target_dir):
            return None

        var_arrays = []
        for var in ECMWF_VARS:
            var_path = os.path.join(target_dir, f"{var}.tif")
            if not os.path.exists(var_path):
                return None
            arr = load_rasterio_file(var_path)
            var_arrays.append(arr)

        frame = np.stack(var_arrays, axis=-1)  # (H, W, 5)
        frames.append(frame)

    data = np.stack(frames, axis=0)  # (T, H, W, 5)

    # Fill NaN per channel using channel means
    valid_means = np.nanmean(data, axis=(0, 1, 2))  # (5,)
    for ch in range(data.shape[-1]):
        fill = float(valid_means[ch])
        if np.isnan(fill) or np.isinf(fill):
            fill = 0.0
        data[..., ch] = np.nan_to_num(data[..., ch], nan=fill, posinf=fill, neginf=fill)

    return data


def build_valid_samples(fwi_dir, ecmwf_dir, fwi_start, fwi_end,
                        encoder_days, lead_start=14, lead_end=46):
    """
    Build list of valid training samples where all data is available.

    Returns:
        List of dicts with keys: issue_date, encoder_dates, target_dates
    """
    print("\n" + "=" * 60)
    print("Building training samples")
    print("=" * 60)

    issue_start = fwi_start - timedelta(days=lead_start)
    issue_end = fwi_end - timedelta(days=lead_end)

    print(f"FWI Target range: {fwi_start.strftime('%Y-%m-%d')} to {fwi_end.strftime('%Y-%m-%d')}")
    print(f"Encoder history window: {encoder_days} days")
    print(f"Forecast lead time: {lead_start}-{lead_end} days")

    fwi_index = build_fwi_index(fwi_dir)

    valid_samples = []
    skipped_counts = {'encoder': 0, 'ecmwf': 0, 'target': 0}

    issue_dates = generate_date_range(issue_start, issue_end)
    print(f"\nChecking {len(issue_dates)} candidate issue dates...")

    for i, issue_date in enumerate(issue_dates, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(issue_dates)}")

        # Check encoder FWI history
        encoder_start = issue_date - timedelta(days=encoder_days)
        encoder_end = issue_date - timedelta(days=1)
        encoder_dates_list = generate_date_range(encoder_start, encoder_end)

        if not all(d in fwi_index for d in encoder_dates_list):
            skipped_counts['encoder'] += 1
            continue

        # Check ECMWF forecasts
        issue_str = issue_date.strftime('%Y%m%d')
        ecmwf_issue_dir = os.path.join(ecmwf_dir, issue_str)
        if not os.path.exists(ecmwf_issue_dir):
            skipped_counts['ecmwf'] += 1
            continue

        # Check target FWI
        target_start = issue_date + timedelta(days=lead_start)
        target_end = issue_date + timedelta(days=lead_end)
        target_dates_list = generate_date_range(target_start, target_end)

        if not all(d in fwi_index for d in target_dates_list):
            skipped_counts['target'] += 1
            continue

        valid_samples.append({
            'issue_date': issue_date,
            'encoder_dates': (encoder_start, encoder_end),
            'target_dates': (target_start, target_end)
        })

    print(f"\nValid samples: {len(valid_samples)} ({len(valid_samples)/max(len(issue_dates),1)*100:.1f}%)")
    print(f"Skipped - missing encoder FWI: {skipped_counts['encoder']}")
    print(f"Skipped - missing ECMWF: {skipped_counts['ecmwf']}")
    print(f"Skipped - missing target FWI: {skipped_counts['target']}")

    return valid_samples


class S2SFWIDataset(Dataset):
    """
    S2S FWI Prediction Dataset.

    Loads FWI history (encoder), ECMWF forecasts (decoder), and target FWI
    on-the-fly with lazy loading and per-channel standardization.

    Args:
        samples: List of sample dicts from build_valid_samples()
        fwi_dir: FWI data directory
        ecmwf_dir: ECMWF data directory
        encoder_days: Number of encoder history days
        patch_size: Patch size for patchify
        lead_start: Forecast lead start (days)
        lead_end: Forecast lead end (days)
    """

    def __init__(self, samples, fwi_dir, ecmwf_dir, encoder_days, patch_size,
                 lead_start=14, lead_end=46):
        self.samples = samples
        self.fwi_dir = fwi_dir
        self.ecmwf_dir = ecmwf_dir
        self.encoder_days = encoder_days
        self.patch_size = patch_size
        self.lead_start = lead_start
        self.lead_end = lead_end
        self.decoder_days = lead_end - lead_start + 1

        self.fwi_index = build_fwi_index(fwi_dir)

        print(f"\nPreprocessing {len(samples)} samples...")
        self._preprocess_all_samples()

    def _preprocess_all_samples(self):
        """Compute standardization parameters from all valid samples."""
        all_fwi_values = []
        all_ecmwf_values = [[] for _ in range(5)]
        valid_samples = []

        for i, sample in enumerate(self.samples):
            if (i + 1) % 50 == 0:
                print(f"  Preprocessing: {i+1}/{len(self.samples)}")
            try:
                encoder_fwi = load_fwi_sequence(
                    self.fwi_index, sample['encoder_dates'][0], sample['encoder_dates'][1]
                )
                if encoder_fwi is None:
                    continue

                decoder_ecmwf = load_ecmwf_sequence(
                    self.ecmwf_dir, sample['issue_date'], self.lead_start, self.lead_end
                )
                if decoder_ecmwf is None:
                    continue

                target_fwi = load_fwi_sequence(
                    self.fwi_index, sample['target_dates'][0], sample['target_dates'][1]
                )
                if target_fwi is None:
                    continue

                valid_fwi = encoder_fwi[np.isfinite(encoder_fwi)]
                if len(valid_fwi) > 0:
                    all_fwi_values.extend(valid_fwi)
                valid_fwi_target = target_fwi[np.isfinite(target_fwi)]
                if len(valid_fwi_target) > 0:
                    all_fwi_values.extend(valid_fwi_target)

                for ch in range(5):
                    valid_ecmwf = decoder_ecmwf[..., ch][np.isfinite(decoder_ecmwf[..., ch])]
                    if len(valid_ecmwf) > 0:
                        all_ecmwf_values[ch].extend(valid_ecmwf)

                valid_samples.append(sample)
            except Exception as e:
                print(f"Warning: sample {sample['issue_date']} failed: {e}")
                continue

        self.samples = valid_samples
        print(f"\nPreprocessing complete. Valid samples: {len(self.samples)}")

        # FWI normalization
        if len(all_fwi_values) > 0:
            self.fwi_mean = float(np.mean(all_fwi_values))
            self.fwi_std = float(np.std(all_fwi_values) + 1e-6)
        else:
            self.fwi_mean, self.fwi_std = 0.0, 1.0
        print(f"[FWI Norm] mean: {self.fwi_mean:.4f}, std: {self.fwi_std:.4f}")

        # ECMWF per-channel normalization
        self.ecmwf_means = []
        self.ecmwf_stds = []
        for ch, var in enumerate(ECMWF_VARS):
            if len(all_ecmwf_values[ch]) > 0:
                mean = float(np.mean(all_ecmwf_values[ch]))
                std = float(np.std(all_ecmwf_values[ch]) + 1e-6)
            else:
                mean, std = 0.0, 1.0
            self.ecmwf_means.append(mean)
            self.ecmwf_stds.append(std)
            print(f"[ECMWF Norm] {var}: mean={mean:.4f}, std={std:.4f}")

        # Determine patch grid from first sample
        if len(self.samples) > 0:
            sample = self.samples[0]
            encoder_fwi = load_fwi_sequence(
                self.fwi_index, sample['encoder_dates'][0], sample['encoder_dates'][1]
            )
            _, self.hw, self.grid = patchify(encoder_fwi, self.patch_size)
            print(f"[Patch Info] Cropped size: {self.hw}, Grid: {self.grid}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoder_fwi = load_fwi_sequence(
            self.fwi_index, sample['encoder_dates'][0], sample['encoder_dates'][1]
        )
        decoder_ecmwf = load_ecmwf_sequence(
            self.ecmwf_dir, sample['issue_date'], self.lead_start, self.lead_end
        )
        target_fwi = load_fwi_sequence(
            self.fwi_index, sample['target_dates'][0], sample['target_dates'][1]
        )

        # Fill NaN
        encoder_fwi = np.nan_to_num(encoder_fwi, nan=self.fwi_mean)
        target_fwi = np.nan_to_num(target_fwi, nan=self.fwi_mean)
        for ch in range(5):
            decoder_ecmwf[..., ch] = np.nan_to_num(
                decoder_ecmwf[..., ch], nan=self.ecmwf_means[ch]
            )

        # Standardize
        encoder_fwi = (encoder_fwi - self.fwi_mean) / self.fwi_std
        target_fwi = (target_fwi - self.fwi_mean) / self.fwi_std
        for ch in range(5):
            decoder_ecmwf[..., ch] = (
                decoder_ecmwf[..., ch] - self.ecmwf_means[ch]
            ) / self.ecmwf_stds[ch]

        # Clip extreme values
        encoder_fwi = np.clip(encoder_fwi, -10, 10)
        decoder_ecmwf = np.clip(decoder_ecmwf, -10, 10)
        target_fwi = np.clip(target_fwi, -10, 10)

        # Patchify
        encoder_patches, _, _ = patchify(encoder_fwi, self.patch_size)
        decoder_patches, _, _ = patchify(decoder_ecmwf, self.patch_size)
        target_patches, _, _ = patchify(target_fwi, self.patch_size)

        return (
            torch.from_numpy(encoder_patches).float(),
            torch.from_numpy(decoder_patches).float(),
            torch.from_numpy(target_patches).float()
        )
