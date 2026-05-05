"""Combine the best parameters from earlier experiments and look at per-image breakdown."""
import csv
import os
import sys
import statistics
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B
from benchmark import TEST_FILES
from experiment import (
    estimate_single_area_log_kde,
    blobs_to_count_with_estimator,
    evaluate,
)


def evaluate_detail(rows, dir_, estimator_fn, sa_factor, name, detect_params=None):
    results = []
    for r in rows:
        path = os.path.join(dir_, r['image_path'])
        truth = int(r['user_count'])
        img = cv2.imread(path)
        if img is None:
            continue
        blobs, _ = B.detect_blobs(img, detect_params)
        total, _ = blobs_to_count_with_estimator(blobs, sa_factor, estimator_fn, detect_params)
        diff = total - truth
        results.append({'image': r['image_path'], 'truth': truth, 'pred': total,
                        'diff': diff, 'rel': abs(diff) / truth if truth > 0 else 0})
    return results


def metrics(rs):
    n = len(rs)
    if n == 0: return None
    abs_errs = [abs(r['diff']) for r in rs]
    rel_errs = [r['rel'] for r in rs]
    diffs = [r['diff'] for r in rs]
    return {
        'mae': sum(abs_errs) / n,
        'rmse': (sum(e * e for e in abs_errs) / n) ** 0.5,
        'mape': sum(rel_errs) / n * 100,
        'bias': sum(diffs) / n,
        'within_5': sum(1 for e in abs_errs if e <= 5) / n * 100,
        'within_10pct': sum(1 for e in rel_errs if e <= 0.10) / n * 100,
        'within_20pct': sum(1 for e in rel_errs if e <= 0.20) / n * 100,
    }


def fmt(m, name):
    return (f"{name:<50} MAE={m['mae']:5.1f}  RMSE={m['rmse']:5.1f}  MAPE={m['mape']:4.1f}%  "
            f"bias={m['bias']:+6.1f}  <=5: {m['within_5']:4.0f}%  <=10%: {m['within_10pct']:4.0f}%  "
            f"<=20%: {m['within_20pct']:4.0f}%")


def main():
    rows = []
    with open(TEST_FILES / 'feedback_rows.csv', 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    print(f"Evaluating on {len(rows)} test images")
    print('-' * 130)

    print("\n=== Combining log-peak + DARK + sa sweep ===")
    best_overall = None
    for dt in [60, 65, 70, 75, 80, 85]:
        for mba in [60, 80, 100]:
            for sa in [0.80, 0.85, 0.90, 0.95, 1.00]:
                p = {'min_blob_area': mba, 'dark_threshold': dt}
                rs = evaluate_detail(rows, str(TEST_FILES), estimate_single_area_log_kde, sa,
                                     f"log-peak, DARK={dt}, MIN_BLOB={mba}, sa={sa:.2f}", p)
                m = metrics(rs)
                name = f"log-peak DARK={dt} MIN_BLOB={mba} sa={sa:.2f}"
                if best_overall is None or m['mae'] < best_overall[0]['mae']:
                    best_overall = (m, name, p, sa, estimate_single_area_log_kde)
                # only print top results
                if m['mae'] < 55 or m['within_10pct'] > 45:
                    print(fmt(m, name))

    print("\n=== Combining current + DARK + sa sweep ===")
    for dt in [60, 65, 70, 75, 80]:
        for mba in [60, 80, 100]:
            for sa in [0.80, 0.85, 0.90, 0.95, 1.00]:
                p = {'min_blob_area': mba, 'dark_threshold': dt}
                rs = evaluate_detail(rows, str(TEST_FILES), B.estimate_single_area, sa,
                                     '', p)
                m = metrics(rs)
                name = f"current  DARK={dt} MIN_BLOB={mba} sa={sa:.2f}"
                if best_overall is None or m['mae'] < best_overall[0]['mae']:
                    best_overall = (m, name, p, sa, B.estimate_single_area)
                if m['mae'] < 52 or m['within_10pct'] > 45:
                    print(fmt(m, name))

    print(f"\n*** BEST overall (by MAE): ***")
    m, name, p, sa, est = best_overall
    print(fmt(m, name))


if __name__ == '__main__':
    main()
