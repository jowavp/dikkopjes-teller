"""
Benchmark script: runs the dikkopjes-teller algorithm (ported from index.html JS)
against all test images and compares to ground-truth user counts from feedback_rows.csv.

Usage: python benchmark.py
"""
import csv
import os
import sys
import argparse
import numpy as np
import cv2

# --- Algorithm constants (mirrors index.html) ---
DARK_THRESHOLD = 75
BIN_BRIGHT_THRESHOLD = 95
DENSITY_BLUR = 201
DENSITY_THRESHOLD = 6
MIN_BLOB_AREA = 80
MIN_SINGLE_AREA = 150.0
CLUMP_RATIO_THRESHOLD = 1.6
LARGE_CLUMP_RATIO = 6.0
LARGE_CLUMP_OVERLAP = 1.4
DEFAULT_SA_FACTOR = 0.80


def find_bin_mask(blurred, params=None):
    p = params or {}
    dark_thr = p.get('dark_threshold', DARK_THRESHOLD)
    bright_thr = p.get('bin_bright_threshold', BIN_BRIGHT_THRESHOLD)
    density_blur = p.get('density_blur', DENSITY_BLUR)
    density_thr = p.get('density_threshold', DENSITY_THRESHOLD)

    _, dark = cv2.threshold(blurred, dark_thr, 255, cv2.THRESH_BINARY_INV)
    density = cv2.GaussianBlur(dark, (density_blur, density_blur), 0)
    _, dense = cv2.threshold(density, density_thr, 255, cv2.THRESH_BINARY)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    if num <= 1:
        return np.full(blurred.shape, 255, dtype=np.uint8)
    # Take all dense components >= 20% of the largest. This is critical for bins
    # where tadpoles cluster in disconnected areas (e.g. left vs right side):
    # taking only the largest component would crop out half the bin.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    keep_threshold = max(1, int(largest_area * 0.20))
    keep_labels = [i + 1 for i, a in enumerate(areas) if a >= keep_threshold]
    region = np.isin(labels, keep_labels).astype(np.uint8) * 255

    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.full(blurred.shape, 255, dtype=np.uint8)

    pts = np.vstack(contours)
    hull = cv2.convexHull(pts)
    mask = np.zeros(blurred.shape, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((20, 20), np.uint8))

    _, bright = cv2.threshold(blurred, bright_thr, 255, cv2.THRESH_BINARY)
    k25 = np.ones((25, 25), np.uint8)
    for _ in range(3):
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k25)
    mask = cv2.bitwise_and(mask, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
    return mask


def detect_blobs(img_bgr, params=None):
    p = params or {}
    dark_thr = p.get('dark_threshold', DARK_THRESHOLD)
    min_blob_area = p.get('min_blob_area', MIN_BLOB_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_mask = find_bin_mask(blurred, params)

    _, dark = cv2.threshold(blurred, dark_thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    blobs = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_blob_area:
            continue
        cx = int(round(centroids[i, 0]))
        cy = int(round(centroids[i, 1]))
        blobs.append({'area': area, 'cx': cx, 'cy': cy})
    return blobs, bin_mask


def estimate_single_area(areas, sa_factor, params=None):
    """
    Robust single-blob area estimator: log-space histogram peak.

    Linear-bin histogram peak (used previously) was very sensitive to small noise
    blobs — when many tiny edge artifacts exist, the peak gets pulled to the floor
    bucket, drastically underestimating the typical tadpole area and inflating clump
    counts. Log-space binning gives equal weight to small and large area regimes
    and is robust to that failure mode.
    """
    p = params or {}
    min_single = p.get('min_single_area', MIN_SINGLE_AREA)
    if not areas:
        return min_single * sa_factor
    if len(areas) < 3:
        return max(min_single, statistics_median(areas)) * sa_factor

    log_areas = np.log(np.asarray(areas, dtype=np.float64))
    hist, edges = np.histogram(log_areas, bins=30)
    peak = int(np.argmax(hist))
    log_peak = (edges[peak] + edges[peak + 1]) / 2.0
    peak_area = float(np.exp(log_peak))
    return max(min_single, peak_area) * sa_factor


def statistics_median(seq):
    s = sorted(seq)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def blobs_to_count(blobs, sa_factor, params=None):
    p = params or {}
    clump_ratio = p.get('clump_ratio_threshold', CLUMP_RATIO_THRESHOLD)
    large_clump_ratio = p.get('large_clump_ratio', LARGE_CLUMP_RATIO)
    large_clump_overlap = p.get('large_clump_overlap', LARGE_CLUMP_OVERLAP)

    if not blobs:
        return 0, [], None
    single_area = estimate_single_area([b['area'] for b in blobs], sa_factor, params)
    detections = []
    total = 0
    for b in blobs:
        ratio = b['area'] / single_area
        if ratio < clump_ratio:
            count = 1
        elif ratio < large_clump_ratio:
            count = round(ratio)
        else:
            count = round(b['area'] / (single_area * large_clump_overlap))
        detections.append({'cx': b['cx'], 'cy': b['cy'], 'count': count, 'area': b['area']})
        total += count
    return total, detections, single_area


def count_tadpoles(img_bgr, sa_factor=DEFAULT_SA_FACTOR, params=None):
    blobs, _ = detect_blobs(img_bgr, params)
    total, detections, single_area = blobs_to_count(blobs, sa_factor, params)
    return total, detections, single_area, blobs


def best_sa_factor_for(blobs, target):
    """Sweep sa_factor 0.60..1.30 and return the factor that best matches target."""
    best = None
    for i in range(71):
        f = 0.60 + i * 0.01
        total, _, _ = blobs_to_count(blobs, f)
        diff = abs(total - target)
        if best is None or diff < best['diff']:
            best = {'factor': round(f, 2), 'predicted': total, 'diff': diff}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='test-files/feedback_rows.csv')
    parser.add_argument('--dir', default='test-files')
    parser.add_argument('--sa-factor', type=float, default=DEFAULT_SA_FACTOR)
    parser.add_argument('--out', default='benchmark_results.csv')
    args = parser.parse_args()

    rows = []
    with open(args.csv, 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    print(f"Running algorithm on {len(rows)} images (sa_factor={args.sa_factor})\n")
    print(f"{'image':<40} {'truth':>6} {'pred':>6} {'diff':>6} {'best_f':>7} {'best_pred':>10}")
    print('-' * 80)

    results = []
    abs_errors = []
    rel_errors = []
    for r in rows:
        path = os.path.join(args.dir, r['image_path'])
        truth = int(r['user_count'])
        img = cv2.imread(path)
        if img is None:
            print(f"  !! kon foto niet lezen: {path}")
            continue
        total, _, single_area, blobs = count_tadpoles(img, args.sa_factor)
        best = best_sa_factor_for(blobs, truth)
        diff = total - truth
        abs_errors.append(abs(diff))
        rel_errors.append(abs(diff) / truth if truth > 0 else 0)
        print(f"{r['image_path']:<40} {truth:>6} {total:>6} {diff:>+6} "
              f"{best['factor']:>7.2f} {best['predicted']:>10}")
        results.append({
            'image': r['image_path'],
            'truth': truth,
            'predicted': total,
            'diff': diff,
            'abs_diff': abs(diff),
            'rel_err': abs(diff) / truth if truth > 0 else 0,
            'best_factor': best['factor'],
            'best_predicted': best['predicted'],
            'best_diff': best['diff'],
            'single_area': round(single_area, 1) if single_area else None,
            'num_blobs': len(blobs),
        })

    print('-' * 80)
    n = len(results)
    if n:
        mae = sum(abs_errors) / n
        rmse = (sum(e * e for e in abs_errors) / n) ** 0.5
        mape = sum(rel_errors) / n * 100
        within_5 = sum(1 for e in abs_errors if e <= 5) / n * 100
        within_10pct = sum(1 for e in rel_errors if e <= 0.10) / n * 100
        bias = sum(r['diff'] for r in results) / n
        print(f"\nMAE  : {mae:.1f}   (mean absolute error)")
        print(f"RMSE : {rmse:.1f}")
        print(f"MAPE : {mape:.1f}%   (mean absolute percentage error)")
        print(f"Bias : {bias:+.1f}   (negative = systematically undercounting)")
        print(f"Within 5 of truth   : {within_5:.0f}%")
        print(f"Within 10% of truth : {within_10pct:.0f}%")

        # Best factor distribution
        from collections import Counter
        factors = Counter(round(r['best_factor'], 2) for r in results)
        print(f"\nBest sa_factor distribution:")
        for f in sorted(factors):
            print(f"  {f:.2f}: {'#' * factors[f]} ({factors[f]})")

    # Write detailed CSV
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nDetailed results: {args.out}")


if __name__ == '__main__':
    main()
