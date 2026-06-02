"""
Benchmark script: runs the dikkopjes-teller algorithm (ported from index.html JS)
against all test images and compares to ground-truth user counts from feedback_rows.csv.

Usage: python benchmark.py
"""
import csv
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES = ROOT / 'test-files'
RESULTS = ROOT / 'benchmark_results'

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
DEFAULT_SA_FACTOR = 0.97

# Resolution-aware scaling. Pixel constants above were calibrated on photos
# in test-files/, which span 0.76 to 1.23 MP. Higher-res photos need them
# scaled — areas linearly with pixel count, kernel/blur sizes with sqrt.
# Reference is set to the top of the calibration range so all calibration
# photos clamp to scale=1.0 (= original behavior preserved exactly).
REFERENCE_PIXELS = 1_250_000


def _odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


# Per-photo adaptive sa_factor models, fit by scripts/analyze_factor_combined.py
# on the combined 88-photo dataset (test-files + test-files2). Top-5 features
# per mode, standardized linear regression.
#
# LOOCV improvement vs fixed defaultSaFactor:
#   standard:   10.38% MAPE -> 9.27%  (-1.11pp)
#   sensitive:   9.76% MAPE -> 7.68%  (-2.08pp)
#
# To retrain: run `python scripts/analyze_factor_combined.py --emit-coefs`
# and paste the JSON below.
ADAPTIVE_MODELS = {
    'standard': {
        # top-5 features; ordered as in the analyze script's correlation rank.
        'features': ['clump_frac', 'cv_area', 'p90_over_median',
                     'largest_over_median', 'std_area'],
        'beta': [0.9972727272727275, 0.05812097426852898, -0.11834816680619258,
                 -0.007631038594075638, 0.10236213376037359, -0.006802614750998628],
        'mu':   [0.4841286306924158, 1.4676699521315202, 4.2325774541524925,
                 23.048469732088286, 5153.453733816543],
        'sd':   [0.059298243352269776, 0.5145865186458295, 1.144921537280663,
                 19.089374099493547, 6387.049687696688],
    },
    'sensitive': {
        # top-5 trained against the global-default counting params. Includes
        # `megapixels` which is what makes high-res photos counted right.
        # Switching to per-mode counting + top-1 dropped megapixels and
        # regressed test-files2; see tune_hyperparams analysis.
        'features': ['clump_frac', 'p90_over_median', 'megapixels',
                     'log_megapixels', 'lh_median'],
        'beta': [1.207386363636364, 0.06980702423311459, -0.025454457849815412,
                 0.07427144671852026, 0.061528563365571096, -0.11766815376801248],
        'mu':   [0.5049657749628903, 2.8431283330512835, 3.2164198295454542,
                 0.8708257852456737, 656.7954545454545],
        'sd':   [0.07460489624099871, 0.5115009115089713, 2.252918576915454,
                 0.7957601720118731, 513.0545372191508],
    },
}
ADAPTIVE_FACTOR_MIN = 0.70
ADAPTIVE_FACTOR_MAX = 1.40


def compute_blob_features(blobs, img_shape):
    """Per-photo features for adaptive sa_factor prediction.

    Must stay in sync with the equivalent function in app.js (computeFeatures)
    and with features_from_blobs() in scripts/analyze_factor_combined.py.
    """
    if not blobs:
        return None
    areas = np.asarray([b['area'] for b in blobs], dtype=np.float64)
    n = len(areas)
    sorted_a = np.sort(areas)
    median = float(np.median(areas))
    half = sorted_a[: max(3, n // 2)]
    lh_median = float(np.median(half))
    largest = float(sorted_a[-1])
    p90 = float(sorted_a[int(0.9 * (n - 1))])
    h, w = img_shape[:2]
    mp = (h * w) / 1_000_000.0
    mean_area = float(areas.mean())
    return {
        'num_blobs': n,
        'median_area': median,
        'lh_median': lh_median,
        'mean_area': mean_area,
        'std_area': float(areas.std()),
        'cv_area': float(areas.std() / mean_area) if mean_area > 0 else 0.0,
        'total_dark': float(areas.sum()),
        'clump_frac': float(np.mean(areas > 1.5 * lh_median)),
        'p90_over_median': p90 / median if median > 0 else 0.0,
        'largest_over_median': largest / median if median > 0 else 0.0,
        'log_density': float(np.log(n / max(1.0, areas.sum()))) if areas.sum() > 0 else 0.0,
        'megapixels': mp,
        'log_megapixels': float(np.log(max(mp, 0.01))),
    }


def predict_adaptive_factor(blobs, img_shape, mode):
    """Predicted sa_factor for this photo + mode, clipped to safe range."""
    if not blobs or mode not in ADAPTIVE_MODELS:
        return DETECTION_MODES.get(mode, {}).get('defaultSaFactor', DEFAULT_SA_FACTOR)
    m = ADAPTIVE_MODELS[mode]
    feats = compute_blob_features(blobs, img_shape)
    x = np.array([feats[f] for f in m['features']])
    xn = (x - np.array(m['mu'])) / np.array(m['sd'])
    z = m['beta'][0] + float(np.dot(m['beta'][1:], xn))
    return max(ADAPTIVE_FACTOR_MIN, min(ADAPTIVE_FACTOR_MAX, z))


def scale_params_for_image(base_params, img):
    """Augment params with resolution-scaled overrides based on image size.

    Modeled after the area constants being absolute pixel counts: a 6 MP photo
    has tadpoles that are ~6× larger in pixel area than the same scene at
    1 MP. Kernel/blur sizes are linear, so they scale by sqrt.
    """
    p = dict(base_params)
    h, w = img.shape[:2]
    area_scale = max(1.0, (h * w) / REFERENCE_PIXELS)
    linear_scale = area_scale ** 0.5

    mode_name = p.get('detection_mode', DEFAULT_DETECTION_MODE)
    mode = DETECTION_MODES[mode_name]

    p.setdefault('area_scale', area_scale)
    p.setdefault('linear_scale', linear_scale)
    p.setdefault('min_blob_area', max(3, int(round(mode['minBlobArea'] * area_scale))))
    p.setdefault('min_single_area', MIN_SINGLE_AREA * area_scale)
    p.setdefault('density_blur', _odd(DENSITY_BLUR * linear_scale))
    p.setdefault('close_kernel', max(3, int(round(3 * linear_scale))))
    p.setdefault('bin_dilate_kernel', max(3, int(round(20 * linear_scale))))
    p.setdefault('bin_bright_close_kernel', max(3, int(round(25 * linear_scale))))
    p.setdefault('bin_final_close_kernel', max(3, int(round(30 * linear_scale))))
    # Bin-mask "dense region" filter. Was 0.20: any density-region smaller
    # than 20% of the largest was dropped. That cuts off small isolated
    # tadpole clusters in box corners (see test-files3 23_100 — 2 tadpoles
    # lost in the bottom-left). Lowered to 0.05 with an absolute pixel
    # floor as a safety against picking up noise. Costs ~0.5pp MAPE on
    # average but markedly improves visual detection coverage.
    p.setdefault('density_keep_ratio', 0.05)
    p.setdefault('density_keep_min_pixels', max(500, int(round(2000 * area_scale))))

    # Per-mode counting hyperparameters (overridable). These don't depend on
    # resolution; they live with scale_params_for_image purely so the per-image
    # params dict ends up self-contained for downstream blobs_to_count calls.
    p.setdefault('clump_ratio_threshold', mode.get('clumpRatioThreshold',
                                                   CLUMP_RATIO_THRESHOLD))
    p.setdefault('large_clump_ratio', mode.get('largeClumpRatio',
                                               LARGE_CLUMP_RATIO))
    p.setdefault('large_clump_overlap', mode.get('largeClumpOverlap',
                                                 LARGE_CLUMP_OVERLAP))
    return p

# Detection modes (mirror app.js DETECTION_MODES). Keep in sync.
DETECTION_MODES = {
    'standard': {
        'darkThreshold': 75,
        'morphClose': True,
        'minBlobArea': 80,
        # Empirical optimum across the combined 88-photo dataset (test-files +
        # test-files2). See scripts/tune_sa_factor.py.
        'defaultSaFactor': 0.98,
        # Counting hyperparameters, jointly tuned per mode by
        # scripts/tune_hyperparams.py (stage 1, adaptive=off):
        #   standard:  10.38% -> 9.29% MAPE on combined (-1.09pp).
        'clumpRatioThreshold': 1.9,
        'largeClumpRatio': 4.0,
        'largeClumpOverlap': 1.20,
    },
    'sensitive': {
        'darkThreshold': 50,
        'morphClose': False,
        'minBlobArea': 30,
        # Empirical optimum across the combined 88-photo dataset.
        'defaultSaFactor': 1.21,
        # Sensitive sticks with the global defaults. tune_hyperparams.py
        # found stage-1 winners (1.8 / 8.0 / 1.30), but in combination with
        # the adaptive regression they regress test-files2 — the lower-
        # complexity adaptive model (top-1) needed for the new counting
        # drops `megapixels` and loses the high-res handling.
        'clumpRatioThreshold': CLUMP_RATIO_THRESHOLD,
        'largeClumpRatio': LARGE_CLUMP_RATIO,
        'largeClumpOverlap': LARGE_CLUMP_OVERLAP,
    },
}
DEFAULT_DETECTION_MODE = 'sensitive'


def find_bin_mask(blurred, params=None):
    p = params or {}
    dark_thr = p.get('dark_threshold', DARK_THRESHOLD)
    bright_thr = p.get('bin_bright_threshold', BIN_BRIGHT_THRESHOLD)
    density_blur = _odd(p.get('density_blur', DENSITY_BLUR))
    density_thr = p.get('density_threshold', DENSITY_THRESHOLD)
    k_dilate = int(p.get('bin_dilate_kernel', 20))
    k_bright = int(p.get('bin_bright_close_kernel', 25))
    k_final = int(p.get('bin_final_close_kernel', 30))

    _, dark = cv2.threshold(blurred, dark_thr, 255, cv2.THRESH_BINARY_INV)
    density = cv2.GaussianBlur(dark, (density_blur, density_blur), 0)
    _, dense = cv2.threshold(density, density_thr, 255, cv2.THRESH_BINARY)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    if num <= 1:
        return np.full(blurred.shape, 255, dtype=np.uint8)
    # Keep dense components down to either `density_keep_ratio` of the largest
    # OR an absolute pixel floor — whichever is smaller — so 2-3 isolated
    # tadpoles in a box corner survive instead of being cropped out.
    keep_ratio = float(p.get('density_keep_ratio', 0.05))
    keep_min_px = int(p.get('density_keep_min_pixels', 2000))
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    keep_threshold = max(1, min(int(largest_area * keep_ratio), keep_min_px))
    keep_labels = [i + 1 for i, a in enumerate(areas) if a >= keep_threshold]
    region = np.isin(labels, keep_labels).astype(np.uint8) * 255

    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.full(blurred.shape, 255, dtype=np.uint8)

    pts = np.vstack(contours)
    hull = cv2.convexHull(pts)
    mask = np.zeros(blurred.shape, dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((k_dilate, k_dilate), np.uint8))

    _, bright = cv2.threshold(blurred, bright_thr, 255, cv2.THRESH_BINARY)
    k_bright_kernel = np.ones((k_bright, k_bright), np.uint8)
    for _ in range(3):
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_bright_kernel)
    mask = cv2.bitwise_and(mask, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_final, k_final), np.uint8))
    return mask


def detect_blobs(img_bgr, params=None):
    p = scale_params_for_image(params or {}, img_bgr)
    mode_name = p.get('detection_mode', DEFAULT_DETECTION_MODE)
    mode = DETECTION_MODES[mode_name]
    dark_thr = p.get('dark_threshold', mode['darkThreshold'])
    min_blob_area = p['min_blob_area']
    morph_close = p.get('morph_close', mode['morphClose'])
    k_close = int(p.get('close_kernel', 3))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_mask = find_bin_mask(blurred, p)

    _, dark = cv2.threshold(blurred, dark_thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    if morph_close:
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                np.ones((k_close, k_close), np.uint8))

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


def detect_blobs_full(img_bgr, params=None):
    """
    Like detect_blobs() but each blob also carries `label` and `bbox`, and the
    function returns the labels matrix so callers can extract per-blob masks
    (needed by the watershed counter).

    Backward-compat wrapper around detect_blobs would have meant changing
    every caller of detect_blobs; this parallel function is cheaper.
    """
    p = scale_params_for_image(params or {}, img_bgr)
    mode_name = p.get('detection_mode', DEFAULT_DETECTION_MODE)
    mode = DETECTION_MODES[mode_name]
    dark_thr = p.get('dark_threshold', mode['darkThreshold'])
    min_blob_area = p['min_blob_area']
    morph_close = p.get('morph_close', mode['morphClose'])
    k_close = int(p.get('close_kernel', 3))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_mask = find_bin_mask(blurred, p)

    _, dark = cv2.threshold(blurred, dark_thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    if morph_close:
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                np.ones((k_close, k_close), np.uint8))

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    blobs = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_blob_area:
            continue
        blobs.append({
            'area': area,
            'cx': int(round(centroids[i, 0])),
            'cy': int(round(centroids[i, 1])),
            'label': i,
            'bbox': (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            ),
        })
    return blobs, bin_mask, labels


def estimate_single_area(areas, sa_factor, params=None):
    """
    Lower-half median: median of the smallest 50% of blob areas.

    Singles dominate the lower half; clumps inflate the upper half. The
    previous log-histogram peak (kept as estimate_single_area_v1_peak below
    for reference) was systematically biased toward the clump cluster on
    photos with heavy clumping.

    Benchmark on 46 ground-truth photos: MAPE 11.4% → 9.8%; within 10%
    of truth 46% → 65%; bias -11.6 → +1.3.
    """
    p = params or {}
    min_single = p.get('min_single_area', MIN_SINGLE_AREA)
    if not areas:
        return min_single * sa_factor
    if len(areas) < 3:
        return max(min_single, statistics_median(areas)) * sa_factor
    sorted_areas = sorted(areas)
    half = sorted_areas[: max(3, len(sorted_areas) // 2)]
    return max(min_single, statistics_median(half)) * sa_factor


def estimate_single_area_v1_peak(areas, sa_factor, params=None):
    """v1 (kept for benchmarking comparison): log-space histogram global peak."""
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


def estimate_single_area_v2(areas, sa_factor, params=None):
    """
    v2: leftmost-mode estimator. Singles are always smaller than clumps, so the
    correct mode is the leftmost substantial peak in the log-area histogram, not
    the global peak. v1 (global peak) was systematically biased toward clumps when
    >50% of blobs were clumps, leading to undercounting (~11% MAPE on benchmark).

    Steps:
      1. Smooth log-area histogram with a 3-bin moving average (denoise jitter).
      2. Find local maxima with at least 30% of the global max bin count.
      3. Return the LEFTMOST such peak (= singles mode).
    """
    p = params or {}
    min_single = p.get('min_single_area', MIN_SINGLE_AREA)
    rel_thresh = p.get('mode_rel_threshold', 0.30)
    if not areas:
        return min_single * sa_factor
    if len(areas) < 3:
        return max(min_single, statistics_median(areas)) * sa_factor

    log_areas = np.log(np.asarray(areas, dtype=np.float64))
    hist, edges = np.histogram(log_areas, bins=30)

    # 3-bin moving average for noise robustness
    smooth = np.convolve(hist, np.ones(3) / 3.0, mode='same')
    threshold = smooth.max() * rel_thresh

    chosen = int(np.argmax(smooth))  # fallback = global peak
    for i in range(len(smooth)):
        if smooth[i] < threshold:
            continue
        left_ok = (i == 0) or smooth[i] >= smooth[i - 1]
        right_ok = (i == len(smooth) - 1) or smooth[i] >= smooth[i + 1]
        if left_ok and right_ok:
            chosen = i
            break

    log_peak = (edges[chosen] + edges[chosen + 1]) / 2.0
    peak_area = float(np.exp(log_peak))
    return max(min_single, peak_area) * sa_factor


def estimate_single_area_v3(areas, sa_factor, params=None):
    """
    v3: lower-half median. The smallest 50% of blobs are dominated by singles;
    the upper half is contaminated by clumps. Taking the median of the lower
    half tracks the true single-area without being pulled up by clumps.

    More robust than v2 (leftmost mode) because it doesn't get fooled by tiny
    noise blobs that form their own small peak.
    """
    p = params or {}
    min_single = p.get('min_single_area', MIN_SINGLE_AREA)
    if not areas:
        return min_single * sa_factor
    if len(areas) < 3:
        return max(min_single, statistics_median(areas)) * sa_factor

    sorted_areas = sorted(areas)
    half = sorted_areas[: max(3, len(sorted_areas) // 2)]
    return max(min_single, statistics_median(half)) * sa_factor


def estimate_single_area_v4(areas, sa_factor, params=None):
    """
    v4: iterative refinement. Start from the global log-histogram peak, then
    classify candidate singles (area < clump_threshold * peak) and recompute
    the peak from those candidates. Converges in 2-3 iterations.

    This corrects for the case where the global peak sits on doubles (heavy
    clumping) by progressively excluding clump-area blobs from the estimation.
    """
    p = params or {}
    min_single = p.get('min_single_area', MIN_SINGLE_AREA)
    clump_ratio = p.get('clump_ratio_threshold', CLUMP_RATIO_THRESHOLD)
    if not areas:
        return min_single * sa_factor
    if len(areas) < 3:
        return max(min_single, statistics_median(areas)) * sa_factor

    log_areas = np.log(np.asarray(areas, dtype=np.float64))

    def peak_of(log_vals, bins=30):
        if len(log_vals) < 3:
            return float(np.exp(np.median(log_vals)))
        hist, edges = np.histogram(log_vals, bins=bins)
        idx = int(np.argmax(hist))
        return float(np.exp((edges[idx] + edges[idx + 1]) / 2.0))

    estimate = peak_of(log_areas)
    for _ in range(3):
        # Candidates = areas plausibly singles (below clump threshold with margin)
        candidate_mask = log_areas < np.log(estimate * clump_ratio)
        candidates = log_areas[candidate_mask]
        if len(candidates) < 5:
            break
        new_estimate = peak_of(candidates, bins=20)
        if abs(new_estimate - estimate) / estimate < 0.02:
            estimate = new_estimate
            break
        estimate = new_estimate

    return max(min_single, estimate) * sa_factor


def statistics_median(seq):
    s = sorted(seq)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def count_blob_watershed(blob, labels, single_area, params=None):
    """
    Count tadpoles in a single blob using a thresholded distance transform.

    For one tadpole approximated as a circle of radius R, the distance transform
    inside that blob peaks at R. For two touching tadpoles, the 'neck' between
    them has small distance, so thresholding the distance transform at some
    fraction of R splits the blob into separate connected components — one per
    tadpole core.

    Returns the number of cores found, with a minimum of 1.
    """
    p = params or {}
    peak_thresh_ratio = p.get('peak_thresh_ratio', 0.45)

    x, y, w, h = blob['bbox']
    # Per-blob mask, padded by 1 pixel so distance transform sees a clean
    # boundary even for blobs that touch the bbox edge
    region = (labels[y:y + h, x:x + w] == blob['label']).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(region, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return 1

    # Tadpole effective radius (equivalent-circle radius for the area)
    radius = float(np.sqrt(single_area / np.pi))
    threshold = peak_thresh_ratio * radius

    _, peaks = cv2.threshold(dist, threshold, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n, _ = cv2.connectedComponents(peaks)
    return max(1, n - 1)  # subtract background label


def count_blob_opening(blob, labels, single_area, params=None):
    """
    Alternative to watershed: morphological opening with an elliptical kernel
    sized to a fraction of the tadpole radius. Opening removes thin features
    (tails) while preserving round bodies. Counts the connected components
    that survive.

    Tadpoles have prominent tails that elongate the blob shape; pure distance-
    transform cores tend to merge along tail-to-body connections. Opening
    eats the tails first, leaving cleanly separated body cores.
    """
    p = params or {}
    open_ratio = p.get('open_ratio', 0.6)  # fraction of single-tadpole radius

    x, y, w, h = blob['bbox']
    region = (labels[y:y + h, x:x + w] == blob['label']).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(region, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)

    radius = float(np.sqrt(single_area / np.pi))
    k = max(3, int(round(open_ratio * radius)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, kernel)
    n, _ = cv2.connectedComponents(opened)
    return max(1, n - 1)


def blobs_to_count_watershed(blobs, labels, sa_factor, params=None):
    """
    Count using watershed-style splitting: small/single blobs counted as 1,
    larger ones split via thresholded distance transform (count = number of
    distance-transform cores).

    Falls back to area-based estimate for blobs where watershed under-counts
    (cores < area/single_area * 0.6) — this catches very tight clumps where
    cores merge into one.
    """
    p = params or {}
    estimator = p.get('estimator', estimate_single_area)
    clump_ratio = p.get('clump_ratio_threshold', CLUMP_RATIO_THRESHOLD)
    use_fallback = p.get('watershed_use_fallback', False)
    fallback_ratio = p.get('watershed_fallback_ratio', 0.6)
    method = p.get('watershed_method', 'dist')  # 'dist' or 'open'
    counter_fn = count_blob_opening if method == 'open' else count_blob_watershed

    if not blobs:
        return 0, [], None
    single_area = estimator([b['area'] for b in blobs], sa_factor, params)

    detections = []
    total = 0
    for b in blobs:
        ratio = b['area'] / single_area
        if ratio < clump_ratio:
            count = 1
        else:
            ws_count = counter_fn(b, labels, single_area, params)
            if use_fallback:
                area_count = max(1, int(round(ratio)))
                # If watershed badly under-counts a very large clump, trust area
                if ws_count < fallback_ratio * area_count:
                    count = area_count
                else:
                    count = ws_count
            else:
                count = ws_count
        detections.append({'cx': b['cx'], 'cy': b['cy'], 'count': count, 'area': b['area']})
        total += count
    return total, detections, single_area


def blobs_to_count_area(blobs, sa_factor, params=None):
    """
    v5 strategy: count = total_dark_area / single_area.

    Skips per-blob clump classification entirely. Empirically the per-photo
    'ideal' single area (total_dark/truth) is much more stable across photos
    than the histogram peak, so this approach is more robust to clumping
    severity. Distributes count proportionally to each blob's area (just for
    visualization — it doesn't affect the total).
    """
    p = params or {}
    estimator = p.get('estimator', estimate_single_area)
    if not blobs:
        return 0, [], None
    single_area = estimator([b['area'] for b in blobs], sa_factor, params)
    total_dark = sum(b['area'] for b in blobs)
    total = max(0, int(round(total_dark / single_area)))
    detections = []
    for b in blobs:
        count = max(1, int(round(b['area'] / single_area)))
        detections.append({'cx': b['cx'], 'cy': b['cy'], 'count': count, 'area': b['area']})
    return total, detections, single_area


def blobs_to_count(blobs, sa_factor, params=None):
    p = params or {}
    clump_ratio = p.get('clump_ratio_threshold', CLUMP_RATIO_THRESHOLD)
    large_clump_ratio = p.get('large_clump_ratio', LARGE_CLUMP_RATIO)
    large_clump_overlap = p.get('large_clump_overlap', LARGE_CLUMP_OVERLAP)
    estimator = p.get('estimator', estimate_single_area)
    counter = p.get('counter', None)
    if counter == 'area':
        return blobs_to_count_area(blobs, sa_factor, params)

    if not blobs:
        return 0, [], None
    single_area = estimator([b['area'] for b in blobs], sa_factor, params)
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


def best_sa_factor_for(blobs, target, params=None):
    """Sweep sa_factor 0.60..1.30 and return the factor that best matches target."""
    best = None
    for i in range(71):
        f = 0.60 + i * 0.01
        total, _, _ = blobs_to_count(blobs, f, params)
        diff = abs(total - target)
        if best is None or diff < best['diff']:
            best = {'factor': round(f, 2), 'predicted': total, 'diff': diff}
    return best


def best_sa_factor_for_watershed(blobs, labels, target, params=None):
    """Same sweep as best_sa_factor_for but using the watershed counter."""
    best = None
    for i in range(71):
        f = 0.60 + i * 0.01
        total, _, _ = blobs_to_count_watershed(blobs, labels, f, params)
        diff = abs(total - target)
        if best is None or diff < best['diff']:
            best = {'factor': round(f, 2), 'predicted': total, 'diff': diff}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(TEST_FILES / 'feedback_rows.csv'))
    parser.add_argument('--dir', default=str(TEST_FILES))
    parser.add_argument('--sa-factor', type=float, default=DEFAULT_SA_FACTOR)
    parser.add_argument('--out', default=str(RESULTS / 'benchmark_results.csv'))
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()),
                        default=DEFAULT_DETECTION_MODE,
                        help='Detection mode: standard (legacy) or sensitive (lower threshold, '
                             'no close, smaller min-area). Mirrors app.js DETECTION_MODES.')
    parser.add_argument('--algo', choices=['v1', 'v2', 'v3', 'v4', 'area', 'watershed'],
                        default='v3',
                        help='v1=global peak (legacy), v2=leftmost mode, v3=lower-half median '
                             '(current production), v4=iterative refinement, '
                             'area=total_dark/single_area (no clump logic), '
                             'watershed=split clumps via distance-transform peaks')
    parser.add_argument('--peak-thresh-ratio', type=float, default=0.45,
                        help='For --algo watershed (dist method): distance-transform threshold '
                             'as fraction of tadpole radius (lower = more permissive splitting)')
    parser.add_argument('--watershed-method', choices=['dist', 'open'], default='dist',
                        help='dist=distance-transform threshold, open=morphological opening')
    parser.add_argument('--open-ratio', type=float, default=0.6,
                        help='For --watershed-method open: ellipse kernel size as fraction of '
                             'tadpole radius')
    parser.add_argument('--single-area', type=float, default=None,
                        help='For --algo area: override estimated single area with a fixed value')
    parser.add_argument('--adaptive', choices=['on', 'off'], default='on',
                        help='on (default): predict sa_factor per photo from blob features. '
                             'off: use the mode default. Forced off if --sa-factor is given.')
    args = parser.parse_args()

    estimators = {
        'v1': estimate_single_area_v1_peak,
        'v2': estimate_single_area_v2,
        'v3': estimate_single_area,  # current production: lower-half median
        'v4': estimate_single_area_v4,
        'area': estimate_single_area,
        'watershed': estimate_single_area,
    }
    params = {
        'estimator': estimators[args.algo],
        'detection_mode': args.detection,
        'peak_thresh_ratio': args.peak_thresh_ratio,
        'watershed_method': args.watershed_method,
        'open_ratio': args.open_ratio,
    }
    # User-given --sa-factor wins over adaptive prediction.
    user_overrode_factor = args.sa_factor != DEFAULT_SA_FACTOR
    if user_overrode_factor:
        adaptive_enabled = False
    else:
        adaptive_enabled = args.adaptive == 'on'
        # Fallback when adaptive is off and user gave no factor: use mode default.
        args.sa_factor = DETECTION_MODES[args.detection]['defaultSaFactor']
    if args.algo == 'area':
        params['counter'] = 'area'
    if args.algo == 'watershed':
        params['counter'] = 'watershed'
    if args.single_area is not None:
        # Override estimator with constant
        const = args.single_area
        params['estimator'] = lambda areas, sa, p=None: const * sa

    rows = []
    with open(args.csv, 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    factor_label = 'adaptive (per-photo)' if adaptive_enabled else f'{args.sa_factor:.2f}'
    print(f"Running algorithm on {len(rows)} images (sa_factor={factor_label})\n")
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
        # Scale pixel constants to this image's resolution. Detection functions
        # also scale internally (idempotent via setdefault), but the estimator's
        # min_single_area floor lives in blobs_to_count and needs scaled params.
        scaled_params = scale_params_for_image(params, img)
        if args.algo == 'watershed':
            blobs, _, labels_mat = detect_blobs_full(img, scaled_params)
            effective_factor = (predict_adaptive_factor(blobs, img.shape, args.detection)
                                if adaptive_enabled else args.sa_factor)
            total, _, single_area = blobs_to_count_watershed(blobs, labels_mat,
                                                             effective_factor, scaled_params)
            best = best_sa_factor_for_watershed(blobs, labels_mat, truth, scaled_params)
        else:
            blobs, _ = detect_blobs(img, scaled_params)
            effective_factor = (predict_adaptive_factor(blobs, img.shape, args.detection)
                                if adaptive_enabled else args.sa_factor)
            total, _, single_area = blobs_to_count(blobs, effective_factor, scaled_params)
            best = best_sa_factor_for(blobs, truth, scaled_params)
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
