"""
Detection redesign experiment.

Hypothese (na watershed-experiment): de telfout zit niet in 'tellen' maar in
'detectie'. De huidige pipeline (dark<75 + MORPH_CLOSE 3x3) plakt naburige
dikkopjes aan elkaar via hun staarten — daardoor moet de teller via clump-
ratios schatten wat eigenlijk apart had moeten zijn.

Doel: probeer detectie-varianten die dikkopjes geSCHEIDEN houden in het
binaire masker. Goede variant => num_blobs / truth nadert 1.0 => de teller
hoeft bijna geen clumps meer te splitsen.

Per variant: voer detectie uit op alle 46 foto's, sweep sa_factor, rapporteer
beste MAPE + n_blobs/truth ratio + bias.

Run: python scripts/experiment_detection.py
"""
import csv
import os
import sys
import statistics
from pathlib import Path
from typing import Callable, List, Dict
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (
    find_bin_mask, blobs_to_count, estimate_single_area,
    DARK_THRESHOLD, MIN_BLOB_AREA, TEST_FILES,
)


# === Detection variants ============================================
# Each function takes the BGR image and returns a list of blob dicts
# {area, cx, cy}. The bin mask is computed once via find_bin_mask().

def _connected_components_to_blobs(mask: np.ndarray, min_area: int) -> List[Dict]:
    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        out.append({
            'area': a,
            'cx': int(round(centroids[i, 0])),
            'cy': int(round(centroids[i, 1])),
        })
    return out


def detect_baseline(img, blurred, bin_mask):
    """Current production pipeline."""
    _, dark = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_no_close(img, blurred, bin_mask):
    """Skip the MORPH_CLOSE — keep tadpoles separated wherever the threshold left a gap."""
    _, dark = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_tight_threshold_60(img, blurred, bin_mask):
    """Lower dark threshold (60) — only catches the darkest body cores."""
    _, dark = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(40, MIN_BLOB_AREA // 2))


def detect_tight_threshold_50(img, blurred, bin_mask):
    """Even tighter threshold (50). Risk: dim tadpoles vanish entirely."""
    _, dark = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(30, MIN_BLOB_AREA // 3))


def detect_otsu(img, blurred, bin_mask):
    """Otsu-thresholded only within the bin region."""
    masked = blurred.copy()
    masked[bin_mask == 0] = 255  # fill outside-bin with white so it doesn't bias Otsu
    # Use only in-bin pixels for histogram
    in_bin = blurred[bin_mask > 0]
    if in_bin.size == 0:
        return []
    otsu_thr, _ = cv2.threshold(in_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(blurred, otsu_thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_adaptive_mean(img, blurred, bin_mask):
    """Adaptive (local-mean) thresholding."""
    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=51, C=10,
    )
    dark = cv2.bitwise_and(adapt, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_adaptive_gaussian(img, blurred, bin_mask):
    """Adaptive (local-Gaussian) thresholding."""
    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=51, C=8,
    )
    dark = cv2.bitwise_and(adapt, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_baseline_then_erode(img, blurred, bin_mask):
    """Baseline + small erosion (3x3 once) before CC. Splits weakly-touching frogs."""
    _, dark = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dark = cv2.erode(dark, np.ones((3, 3), np.uint8), iterations=1)
    return _connected_components_to_blobs(dark, max(40, MIN_BLOB_AREA // 2))


def detect_no_close_then_erode(img, blurred, bin_mask):
    """No close, plus erosion to break thin tail-bridges."""
    _, dark = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    dark = cv2.erode(dark, np.ones((3, 3), np.uint8), iterations=1)
    return _connected_components_to_blobs(dark, max(40, MIN_BLOB_AREA // 2))


def detect_tight60_no_close(img, blurred, bin_mask):
    """Combine tighter threshold + no close."""
    _, dark = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(40, MIN_BLOB_AREA // 2))


def detect_tight65_open3(img, blurred, bin_mask):
    """Tight threshold + opening 3x3 (removes tail-bridges, less aggressive than close)."""
    _, dark = cv2.threshold(blurred, 65, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _connected_components_to_blobs(dark, max(40, MIN_BLOB_AREA // 2))


def detect_tight45(img, blurred, bin_mask):
    """Even tighter than 50 — only the very darkest body cores."""
    _, dark = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(25, MIN_BLOB_AREA // 4))


def detect_tight40(img, blurred, bin_mask):
    """Stricter still."""
    _, dark = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(20, MIN_BLOB_AREA // 4))


def detect_tight50_minarea30(img, blurred, bin_mask):
    """tight50 with smaller min-area; tighter cores are smaller."""
    _, dark = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, 30)


def detect_relative_to_bin_median(img, blurred, bin_mask):
    """
    Threshold = median grey within bin - 40. Adapts per-photo to the bin's
    actual brightness, instead of a hard-coded 75. Still rejects pixels that
    are merely 'darker than average' — only the darkest fraction.
    """
    in_bin = blurred[bin_mask > 0]
    if in_bin.size == 0:
        return []
    bin_median = int(np.median(in_bin))
    thr = max(20, bin_median - 40)
    _, dark = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(40, MIN_BLOB_AREA // 2))


def detect_relative_minus60(img, blurred, bin_mask):
    """Even more relative shift: bin_median - 60."""
    in_bin = blurred[bin_mask > 0]
    if in_bin.size == 0:
        return []
    bin_median = int(np.median(in_bin))
    thr = max(20, bin_median - 60)
    _, dark = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(30, MIN_BLOB_AREA // 3))


def detect_tight55(img, blurred, bin_mask):
    """Between tight50 and tight60."""
    _, dark = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, max(30, MIN_BLOB_AREA // 2))


def detect_tight50_minarea60(img, blurred, bin_mask):
    """tight50 but min_area = 60 to avoid fragmenting single tadpoles."""
    _, dark = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, 60)


def detect_tight50_minarea80(img, blurred, bin_mask):
    """tight50 but min_area = 80 (same as baseline MIN_BLOB_AREA)."""
    _, dark = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, 80)


def detect_tight60_minarea80(img, blurred, bin_mask):
    """tight60 but min_area = 80."""
    _, dark = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, 80)


def _percentile_threshold(blurred, bin_mask, pct):
    in_bin = blurred[bin_mask > 0]
    if in_bin.size == 0:
        return None
    return int(np.percentile(in_bin, pct))


def detect_pct15(img, blurred, bin_mask):
    """Threshold at 15th percentile of bin grayscale (darkest 15%)."""
    thr = _percentile_threshold(blurred, bin_mask, 15)
    if thr is None:
        return []
    _, dark = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_pct20(img, blurred, bin_mask):
    """Threshold at 20th percentile."""
    thr = _percentile_threshold(blurred, bin_mask, 20)
    if thr is None:
        return []
    _, dark = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_pct25(img, blurred, bin_mask):
    """Threshold at 25th percentile."""
    thr = _percentile_threshold(blurred, bin_mask, 25)
    if thr is None:
        return []
    _, dark = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, MIN_BLOB_AREA)


def detect_pct20_minarea50(img, blurred, bin_mask):
    """Threshold at 20th percentile + smaller min area for tight detection."""
    thr = _percentile_threshold(blurred, bin_mask, 20)
    if thr is None:
        return []
    _, dark = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bin_mask)
    return _connected_components_to_blobs(dark, 50)


def detect_two_stage(img, blurred, bin_mask):
    """
    Two-stage: baseline gives 'tadpole regions'; within those, take only the
    very darkest pixels (relative to local mean inside the region).
    """
    # Stage 1: classic baseline mask, gives the "tadpole zones"
    _, dark1 = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark1 = cv2.bitwise_and(dark1, bin_mask)
    dark1 = cv2.morphologyEx(dark1, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Stage 2: within those zones, threshold tighter
    _, dark2 = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark1, dark2)
    return _connected_components_to_blobs(dark, max(30, MIN_BLOB_AREA // 3))


VARIANTS: Dict[str, Callable] = {
    'baseline': detect_baseline,
    'no_close': detect_no_close,
    'tight60': detect_tight_threshold_60,
    'tight50': detect_tight_threshold_50,
    'otsu': detect_otsu,
    'adaptive_mean': detect_adaptive_mean,
    'adaptive_gauss': detect_adaptive_gaussian,
    'erode_3x3': detect_baseline_then_erode,
    'no_close_erode': detect_no_close_then_erode,
    'tight60_no_close': detect_tight60_no_close,
    'tight65_open3': detect_tight65_open3,
    'tight45': detect_tight45,
    'tight40': detect_tight40,
    'tight55': detect_tight55,
    'tight50_minarea30': detect_tight50_minarea30,
    'tight50_minarea60': detect_tight50_minarea60,
    'tight50_minarea80': detect_tight50_minarea80,
    'tight60_minarea80': detect_tight60_minarea80,
    'pct15': detect_pct15,
    'pct20': detect_pct20,
    'pct25': detect_pct25,
    'pct20_minarea50': detect_pct20_minarea50,
    'rel_median-40': detect_relative_to_bin_median,
    'rel_median-60': detect_relative_minus60,
    'two_stage': detect_two_stage,
}


# === Evaluation =======================================================

def load_rows():
    rows = []
    with open(TEST_FILES / 'feedback_rows.csv', 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)
    return rows


def run_variant(name, fn, rows, sa_factors):
    """For one detection variant, sweep sa_factor and return best result."""
    # Cache blobs per image — sweep is then just ratio counting
    cache = []
    for r in rows:
        path = TEST_FILES / r['image_path']
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        bin_mask = find_bin_mask(blurred)
        blobs = fn(img, blurred, bin_mask)
        cache.append({'blobs': blobs, 'truth': int(r['user_count'])})

    # Pure detection metric: how many blobs vs truth on average?
    bpr = [c['blobs'] and (len(c['blobs']) / c['truth']) or 0 for c in cache]
    mean_bpr = float(np.mean(bpr))

    # Sweep sa_factor; pick best by MAPE
    best = None
    for sa in sa_factors:
        errs, rels, biases = [], [], []
        for c in cache:
            total, _, _ = blobs_to_count(c['blobs'], sa)
            d = total - c['truth']
            errs.append(abs(d))
            rels.append(abs(d) / c['truth'] if c['truth'] > 0 else 0)
            biases.append(d)
        mape = float(np.mean(rels)) * 100
        mae = float(np.mean(errs))
        bias = float(np.mean(biases))
        within10 = float(np.mean([1 if r <= 0.10 else 0 for r in rels])) * 100
        if best is None or mape < best['mape']:
            best = {
                'sa': sa, 'mape': mape, 'mae': mae, 'bias': bias,
                'within10': within10,
            }
    best['mean_blobs_per_truth'] = mean_bpr
    best['n_photos'] = len(cache)
    return best


def main():
    rows = load_rows()
    print(f"Evaluating {len(VARIANTS)} detection variants on {len(rows)} photos.\n")
    sa_factors = [round(x * 0.05, 2) for x in range(14, 27)]  # 0.70 .. 1.30

    print(f"{'variant':<22} {'best_sa':>8} {'MAPE':>7} {'MAE':>6} "
          f"{'bias':>7} {'within10':>9} {'blobs/truth':>12}")
    print('-' * 78)
    results = {}
    for name, fn in VARIANTS.items():
        try:
            r = run_variant(name, fn, rows, sa_factors)
        except Exception as e:
            print(f"{name:<22}  ERROR: {e}")
            continue
        results[name] = r
        print(f"{name:<22} {r['sa']:>8.2f} {r['mape']:>6.1f}% {r['mae']:>6.1f} "
              f"{r['bias']:>+7.1f} {r['within10']:>8.0f}% {r['mean_blobs_per_truth']:>12.2f}")

    print('-' * 78)
    if results:
        best = min(results.items(), key=lambda kv: kv[1]['mape'])
        print(f"\nBest by MAPE   : {best[0]} ({best[1]['mape']:.1f}%, "
              f"baseline {results.get('baseline', {}).get('mape', '?'):.1f}%)")
        best_w = max(results.items(), key=lambda kv: kv[1]['within10'])
        print(f"Best by within10: {best_w[0]} ({best_w[1]['within10']:.0f}%, "
              f"baseline {results.get('baseline', {}).get('within10', '?'):.0f}%)")

        # Per-photo regression check: pick top contender, see how many photos
        # got better vs worse compared to baseline
        contender_name = best[0] if best[0] != 'baseline' else None
        if contender_name:
            print(f"\nPer-photo win/loss vs baseline (using {contender_name} at sa={results[contender_name]['sa']:.2f}):")
            sa_b = results['baseline']['sa']
            sa_c = results[contender_name]['sa']
            cache_b = []
            cache_c = []
            for r in rows:
                path = TEST_FILES / r['image_path']
                img = cv2.imread(str(path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                bin_mask = find_bin_mask(blurred)
                bb = VARIANTS['baseline'](img, blurred, bin_mask)
                cb = VARIANTS[contender_name](img, blurred, bin_mask)
                truth = int(r['user_count'])
                tb, _, _ = blobs_to_count(bb, sa_b)
                tc, _, _ = blobs_to_count(cb, sa_c)
                cache_b.append((r['image_path'], truth, tb))
                cache_c.append((r['image_path'], truth, tc))
            wins = ties = losses = 0
            for (n, t, b), (_, _, c) in zip(cache_b, cache_c):
                eb = abs(b - t)
                ec = abs(c - t)
                if ec < eb: wins += 1
                elif ec == eb: ties += 1
                else: losses += 1
            print(f"  wins   : {wins:>3}/{len(cache_b)}")
            print(f"  ties   : {ties:>3}/{len(cache_b)}")
            print(f"  losses : {losses:>3}/{len(cache_b)}")


if __name__ == '__main__':
    main()
