#!/usr/bin/env python3
"""
Per-year Lift@5000 for any save_window_scores dir. CLI version of the
inline analysis we did for 4y enc14, generalized.

Usage:
  python scripts/per_year_lift.py \
    --scores_dir outputs/window_scores_full/v3_9ch_enc14_2000 \
    --tag 22y_enc14
"""
import argparse, glob, os, re, sys
from collections import defaultdict
import numpy as np


def parse_date(fname):
    m = re.search(r'window_\d+_(\d{4})-(\d{2})-(\d{2})\.npz$', fname)
    if not m: return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores_dir', required=True)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--k', type=int, default=5000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.scores_dir, 'window_*.npz')))
    print(f'  {len(files)} window files')

    by_year = defaultdict(list)
    for f in files:
        d = parse_date(os.path.basename(f))
        if not d: continue
        y, _, _ = d
        npz = np.load(f)
        prob = npz['prob_agg'].astype(np.float32).ravel()
        label = npz['label_agg'].astype(np.uint8).ravel()
        n_fire = int(label.sum())
        if n_fire == 0:
            by_year[y].append(None); continue
        K = args.k
        idx = np.argpartition(-prob, K-1)[:K]
        n_pos = int(label[idx].sum())
        base = float(label.mean())
        lift = (n_pos / K) / base if base > 0 else float('nan')
        prec = n_pos / K
        by_year[y].append((lift, prec, n_fire))

    print()
    print(f'TAG: {args.tag}')
    print(f'{"year":>5} {"n_win":>5} {"with_fire":>10} {"total_fire_px":>14} {"lift_5k":>8} {"prec_5k":>8}')
    for y in sorted(by_year.keys()):
        entries = by_year[y]
        wf = [e for e in entries if e is not None]
        if not wf: continue
        lifts = [e[0] for e in wf]
        precs = [e[1] for e in wf]
        fires = sum(e[2] for e in wf)
        print(f'{y:>5} {len(entries):>5} {len(wf):>10} {fires:>14,d} {np.mean(lifts):>8.3f} {np.mean(precs):>8.3f}')


if __name__ == '__main__':
    main()
