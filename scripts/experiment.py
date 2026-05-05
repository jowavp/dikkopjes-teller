"""
Experiment with different parameters / single_area estimators to find what improves accuracy.

Compares the current algorithm against several variants on the full test set.
"""
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


def estimate_single_area_median(areas, sa_factor, params=None):
    """Estimator V2: filter small noise, then take median of remaining blobs."""
    p = params or {}
    min_single = p.get('min_single_area', 150.0)
    noise_cutoff = p.get('noise_cutoff', 80)

    if not areas:
        return min_single * sa_factor
    filtered = [a for a in areas if a >= noise_cutoff]
    if not filtered:
        return min_single * sa_factor
    med = statistics.median(filtered)
    return max(min_single, med) * sa_factor


def estimate_single_area_trimmed(areas, sa_factor, params=None):
    """Estimator V3: take median of bottom 60% (excludes large clumps that pull up the median)."""
    p = params or {}
    min_single = p.get('min_single_area', 150.0)
    noise_cutoff = p.get('noise_cutoff', 80)

    if not areas:
        return min_single * sa_factor
    filtered = sorted(a for a in areas if a >= noise_cutoff)
    if not filtered:
        return min_single * sa_factor
    # Take the 25th-65th percentile range mean — represents typical singles
    lo = int(len(filtered) * 0.25)
    hi = int(len(filtered) * 0.65)
    if hi <= lo:
        med = filtered[len(filtered) // 2]
    else:
        med = statistics.median(filtered[lo:hi])
    return max(min_single, med) * sa_factor


def estimate_single_area_log_kde(areas, sa_factor, params=None):
    """Estimator V4: histogram peak in log-space (more robust than linear bins)."""
    p = params or {}
    min_single = p.get('min_single_area', 150.0)
    noise_cutoff = p.get('noise_cutoff', 80)

    if not areas:
        return min_single * sa_factor
    filtered = [a for a in areas if a >= noise_cutoff]
    if len(filtered) < 3:
        return min_single * sa_factor
    log_areas = np.log(filtered)
    # Histogram in log space, 30 bins
    hist, edges = np.histogram(log_areas, bins=30)
    peak = int(np.argmax(hist))
    log_peak = (edges[peak] + edges[peak + 1]) / 2
    peak_area = float(np.exp(log_peak))
    return max(min_single, peak_area) * sa_factor


def blobs_to_count_with_estimator(blobs, sa_factor, estimator, params=None):
    p = params or {}
    if not blobs:
        return 0, []
    single_area = estimator([b['area'] for b in blobs], sa_factor, p)
    detections = []
    total = 0
    for b in blobs:
        ratio = b['area'] / single_area
        if ratio < B.CLUMP_RATIO_THRESHOLD:
            count = 1
        elif ratio < B.LARGE_CLUMP_RATIO:
            count = round(ratio)
        else:
            count = round(b['area'] / (single_area * B.LARGE_CLUMP_OVERLAP))
        detections.append({'cx': b['cx'], 'cy': b['cy'], 'count': count})
        total += count
    return total, detections


def evaluate(rows, dir_, estimator_fn, sa_factor, name, detect_params=None):
    abs_errs = []
    rel_errs = []
    diffs = []
    for r in rows:
        path = os.path.join(dir_, r['image_path'])
        truth = int(r['user_count'])
        img = cv2.imread(path)
        if img is None:
            continue
        blobs, _ = B.detect_blobs(img, detect_params)
        total, _ = blobs_to_count_with_estimator(blobs, sa_factor, estimator_fn, detect_params)
        diff = total - truth
        abs_errs.append(abs(diff))
        rel_errs.append(abs(diff) / truth if truth > 0 else 0)
        diffs.append(diff)
    n = len(abs_errs)
    if n == 0:
        return None
    mae = sum(abs_errs) / n
    mape = sum(rel_errs) / n * 100
    bias = sum(diffs) / n
    within_5 = sum(1 for e in abs_errs if e <= 5) / n * 100
    within_10pct = sum(1 for e in rel_errs if e <= 0.10) / n * 100
    rmse = (sum(e * e for e in abs_errs) / n) ** 0.5
    print(f"{name:<40} MAE={mae:5.1f}  RMSE={rmse:5.1f}  MAPE={mape:4.1f}%  bias={bias:+6.1f}  "
          f"<=5: {within_5:4.0f}%  <=10%: {within_10pct:4.0f}%")
    return {'mae': mae, 'mape': mape, 'within_5': within_5, 'within_10pct': within_10pct, 'rmse': rmse}


def main():
    rows = []
    with open(TEST_FILES / 'feedback_rows.csv', 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    print(f"Evaluating on {len(rows)} test images\n")
    print(f"{'variant':<40} {'metrics':<70}")
    print('-' * 110)

    # Baseline (current algorithm)
    evaluate(rows, str(TEST_FILES), B.estimate_single_area, 0.95,
             "current (histogram peak), sa=0.95")

    # New estimators
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 0.95,
             "median (cutoff=80), sa=0.95")
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.00,
             "median (cutoff=80), sa=1.00")
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.10,
             "median (cutoff=80), sa=1.10")
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.20,
             "median (cutoff=80), sa=1.20")
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.30,
             "median (cutoff=80), sa=1.30")

    evaluate(rows, str(TEST_FILES), estimate_single_area_trimmed, 0.95,
             "trimmed mean (25-65%), sa=0.95")
    evaluate(rows, str(TEST_FILES), estimate_single_area_trimmed, 1.10,
             "trimmed mean (25-65%), sa=1.10")
    evaluate(rows, str(TEST_FILES), estimate_single_area_trimmed, 1.20,
             "trimmed mean (25-65%), sa=1.20")

    evaluate(rows, str(TEST_FILES), estimate_single_area_log_kde, 0.95,
             "log-space hist peak, sa=0.95")
    evaluate(rows, str(TEST_FILES), estimate_single_area_log_kde, 1.00,
             "log-space hist peak, sa=1.00")
    evaluate(rows, str(TEST_FILES), estimate_single_area_log_kde, 1.10,
             "log-space hist peak, sa=1.10")

    # Try with stricter MIN_BLOB_AREA in detect_blobs
    print("\n--- with MIN_BLOB_AREA=80 (stricter noise filter) ---")
    p80 = {'min_blob_area': 80}
    evaluate(rows, str(TEST_FILES), B.estimate_single_area, 0.95,
             "current, MIN_BLOB=80, sa=0.95", p80)
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 0.95,
             "median, MIN_BLOB=80, sa=0.95", p80)
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.00,
             "median, MIN_BLOB=80, sa=1.00", p80)
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.10,
             "median, MIN_BLOB=80, sa=1.10", p80)
    evaluate(rows, str(TEST_FILES), estimate_single_area_median, 1.20,
             "median, MIN_BLOB=80, sa=1.20", p80)


if __name__ == '__main__':
    main()
