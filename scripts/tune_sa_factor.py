"""
Sweep `defaultSaFactor` per detection mode against the COMBINED ground-truth
set (test-files + test-files2) and report the optimum.

Detection is expensive but sa_factor evaluation is cheap, so we detect blobs
once per photo and replay across the sweep.

Usage:
  python scripts/tune_sa_factor.py
  python scripts/tune_sa_factor.py --detection sensitive
"""
import argparse
import csv
import os
from pathlib import Path

import cv2

from benchmark import (
    DETECTION_MODES,
    TEST_FILES,
    blobs_to_count,
    detect_blobs,
    estimate_single_area,
    scale_params_for_image,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES2 = ROOT / 'test-files2'


def load_dataset(csv_path, image_dir):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        for r in csv.DictReader(f):
            if not r.get('image_path'):
                continue
            rows.append((image_dir / r['image_path'], int(r['user_count'])))
    return rows


def detect_all(dataset, detection_mode):
    """Run detect_blobs once per photo; return (blobs, scaled_params, truth) per row."""
    base_params = {'detection_mode': detection_mode, 'estimator': estimate_single_area}
    out = []
    for path, truth in dataset:
        img = cv2.imread(str(path))
        if img is None:
            print(f'  !! kon foto niet lezen: {path}')
            continue
        scaled = scale_params_for_image(base_params, img)
        blobs, _ = detect_blobs(img, scaled)
        out.append({'path': path, 'truth': truth, 'blobs': blobs, 'params': scaled})
    return out


def evaluate(cached, sa_factor):
    """Replay blobs_to_count across cached detections for a given sa_factor."""
    abs_err = 0.0
    rel_err = 0.0
    within10 = 0
    bias = 0
    n = len(cached)
    for c in cached:
        total, _, _ = blobs_to_count(c['blobs'], sa_factor, c['params'])
        diff = total - c['truth']
        abs_err += abs(diff)
        rel_err += abs(diff) / c['truth'] if c['truth'] > 0 else 0
        if c['truth'] > 0 and abs(diff) / c['truth'] <= 0.10:
            within10 += 1
        bias += diff
    return {
        'mae': abs_err / n,
        'mape': rel_err / n * 100,
        'within10': within10 / n * 100,
        'bias': bias / n,
    }


def sweep_factor(cached, label, lo=0.80, hi=1.40, step=0.01):
    print(f'\n=== {label} (n={len(cached)}) ===')
    print(f'{"factor":>7} {"MAPE":>7} {"MAE":>7} {"w/10%":>7} {"bias":>7}')
    best = None
    f = lo
    results = []
    while f <= hi + 1e-9:
        m = evaluate(cached, f)
        results.append((round(f, 2), m))
        if best is None or m['mape'] < best[1]['mape']:
            best = (round(f, 2), m)
        f += step
    # Print a few notable rows + the best
    for factor, m in results:
        if factor in (lo, 1.00, 1.20, 1.30, best[0], hi):
            mark = '  <-- best' if factor == best[0] else ''
            print(f'{factor:>7.2f} {m["mape"]:>6.2f}% {m["mae"]:>7.1f} '
                  f'{m["within10"]:>6.0f}% {m["bias"]:>+7.1f}{mark}')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()),
                        default=None,
                        help='Tune just one mode (default: both)')
    args = parser.parse_args()
    modes = [args.detection] if args.detection else list(DETECTION_MODES.keys())

    ds_old = load_dataset(TEST_FILES / 'feedback_rows.csv', TEST_FILES)
    ds_new = load_dataset(TEST_FILES2 / 'ground_truth.csv', TEST_FILES2)
    print(f'Loaded {len(ds_old)} old + {len(ds_new)} new = {len(ds_old) + len(ds_new)} photos')

    for mode in modes:
        print(f'\n###### MODE: {mode} ######')
        cached_old = detect_all(ds_old, mode)
        cached_new = detect_all(ds_new, mode)
        cached_all = cached_old + cached_new
        sweep_factor(cached_old, f'{mode} / test-files only')
        sweep_factor(cached_new, f'{mode} / test-files2 only')
        sweep_factor(cached_all, f'{mode} / COMBINED')


if __name__ == '__main__':
    main()
