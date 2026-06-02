"""
Feasibility test: can we use the detected bin-mask area as an absolute
scale reference for tadpole size?

Hypothesis: if every photo is of the same physical box, then
  single_tadpole_pixels = K * bin_mask_pixels
with K being a stable constant (regardless of camera angle, distance,
resolution).

For each ground-truth photo we measure:
  - bin_area_pixels = sum of pixels in the bin mask from findBinMask
  - sum_blob_pixels = total pixels classified as tadpoles
  - true_single_pixels = sum_blob_pixels / true_count
  - ratio K = true_single_pixels / bin_area_pixels

Then we report the coefficient of variation (std/mean) of K. Low CV
means the bin-area reference works; high CV means perspective/angle
noise dominates and we'd need rectification.

We also do a counterfactual: predict count using
  single_area = median(K) * bin_area_pixels
and compare MAPE against current production (adaptive regression).

Usage:
  python scripts/analyze_bin_scale.py
  python scripts/analyze_bin_scale.py --detection sensitive
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (
    DETECTION_MODES,
    TEST_FILES,
    blobs_to_count,
    detect_blobs,
    scale_params_for_image,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES2 = ROOT / 'test-files2'


def detect_with_bin(img, params):
    """Run detect_blobs but also capture the bin mask area.

    detect_blobs already computes find_bin_mask internally and returns it
    as the second tuple element.
    """
    blobs, bin_mask = detect_blobs(img, params)
    bin_area = int((bin_mask > 0).sum())
    return blobs, bin_area


def collect(detection_mode):
    rows = []
    for csv_path, image_dir, label in [
        (TEST_FILES / 'feedback_rows.csv', TEST_FILES, 'test-files'),
        (TEST_FILES2 / 'ground_truth.csv', TEST_FILES2, 'test-files2'),
    ]:
        with open(csv_path, 'r', newline='') as f:
            for r in csv.DictReader(f):
                if not r.get('image_path'):
                    continue
                path = image_dir / r['image_path']
                img = cv2.imread(str(path))
                if img is None:
                    continue
                truth = int(r['user_count'])
                p = scale_params_for_image({'detection_mode': detection_mode}, img)
                blobs, bin_area = detect_with_bin(img, p)
                if truth <= 0 or bin_area <= 0 or not blobs:
                    continue
                sum_blob = sum(b['area'] for b in blobs)
                true_single = sum_blob / truth
                rows.append({
                    'path': str(path),
                    'name': Path(path).name,
                    'folder': label,
                    'truth': truth,
                    'bin_area': bin_area,
                    'sum_blob': sum_blob,
                    'true_single': true_single,
                    'ratio_K': true_single / bin_area,
                    'mp': (img.shape[0] * img.shape[1]) / 1e6,
                    'blobs': blobs,
                    'params': p,
                })
    return rows


def stats(arr, label):
    arr = np.asarray(arr, dtype=float)
    print(f'  {label}: mean={arr.mean():.4e}  std={arr.std():.4e}  '
          f'CV={arr.std()/abs(arr.mean())*100:.2f}%  '
          f'range=[{arr.min():.4e}, {arr.max():.4e}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()),
                        default='sensitive')
    args = parser.parse_args()

    print(f'### detection mode: {args.detection}\n')
    rows = collect(args.detection)
    print(f'Collected {len(rows)} photos\n')

    K_all = [r['ratio_K'] for r in rows]
    K_old = [r['ratio_K'] for r in rows if r['folder'] == 'test-files']
    K_new = [r['ratio_K'] for r in rows if r['folder'] == 'test-files2']

    print('Ratio K = true_single_pixels / bin_area_pixels:')
    stats(K_all, 'combined ')
    stats(K_old, 'test-files')
    stats(K_new, 'test-files2')

    # If CV is < 15% the hypothesis is supported. If 15-30% perspective
    # noise is meaningful but K is still usable as a prior. >30%: too noisy.
    cv_combined = float(np.std(K_all) / np.mean(K_all)) * 100
    print(f'\nVerdict: combined CV = {cv_combined:.1f}%')
    if cv_combined < 15:
        print('  -> Strong signal. Bin-area scale is a reliable reference.')
    elif cv_combined < 30:
        print('  -> Workable. Could improve estimator but not eliminate noise.')
    else:
        print('  -> Too noisy. Perspective/angle dominates over scale.')

    # Per-folder K medians (the constant we would ship)
    K_med_combined = float(np.median(K_all))
    K_med_old = float(np.median(K_old))
    K_med_new = float(np.median(K_new))
    print(f'\nMedian K values: combined={K_med_combined:.4e} '
          f'old={K_med_old:.4e} new={K_med_new:.4e}')

    # ---- Counterfactual: predict count using K * bin_area as single_area ----
    print('\nCounterfactual: single_area = K_med * bin_area_pixels')
    for label, K in [('K_combined', K_med_combined),
                     ('K_test-files', K_med_old),
                     ('K_test-files2', K_med_new)]:
        errs_old, errs_new = [], []
        for r in rows:
            single = K * r['bin_area']
            p = dict(r['params'])
            # Skip the estimator entirely; inject our single area as the floor
            # AND zero out sa_factor's lower-half-median dance by providing a
            # constant. The cleanest way is to bypass blobs_to_count's
            # estimator. blobs_to_count reads `min_single_area` from params and
            # then maxes it with the lower-half median, then multiplies by
            # sa_factor. We want pure single = K*bin, so set min_single_area
            # huge AND sa_factor=1: the max() makes single = min_single_area.
            p['min_single_area'] = single
            # Override estimator to a constant returning single (so the lower-
            # half median doesn't kick in either).
            p['estimator'] = lambda areas, sa, p=None, s=single: s * sa
            total, _, _ = blobs_to_count(r['blobs'], 1.0, p)
            diff = abs(total - r['truth']) / r['truth']
            if r['folder'] == 'test-files':
                errs_old.append(diff)
            else:
                errs_new.append(diff)
        mape_old = np.mean(errs_old) * 100
        mape_new = np.mean(errs_new) * 100
        mape_all = np.mean(errs_old + errs_new) * 100
        print(f'  {label:<16}: combined MAPE {mape_all:.2f}%  '
              f'(test-files {mape_old:.2f}%, test-files2 {mape_new:.2f}%)')

    # ---- Per-photo K distribution check ----
    print('\nPer-photo K (sorted, every 8th entry):')
    sorted_rows = sorted(rows, key=lambda r: r['ratio_K'])
    for i in range(0, len(sorted_rows), max(1, len(sorted_rows) // 12)):
        r = sorted_rows[i]
        print(f'  K={r["ratio_K"]:.4e}  bin={r["bin_area"]:>10,d}  '
              f'truth={r["truth"]:>4d}  blobs={len(r["blobs"]):>4d}  '
              f'mp={r["mp"]:.2f}  {r["folder"]:<11} {r["name"]}')


if __name__ == '__main__':
    main()
