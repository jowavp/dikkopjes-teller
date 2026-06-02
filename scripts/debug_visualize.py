"""
Debug: visualize the algorithm pipeline for problem images, matching the
production app: resolution-aware scaling + adaptive sa_factor + per-mode
counting params + per-mode adaptive regression.

Saves the same overlay the app shows + the intermediate bin mask and dark
mask, plus a per-blob diagnostic CSV that lists every detection with its
area, area/single_area ratio, and predicted count — so you can scan for
"single classified as small clump" or "small blob classified as single"
errors without eyeballing every circle.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B
from benchmark import DETECTION_MODES, ROOT, TEST_FILES


def visualize(img_bgr, mode, sa_factor_override, truth, out_prefix):
    h, w = img_bgr.shape[:2]
    # Match production: scale params for this image's resolution + mode.
    params = B.scale_params_for_image({'detection_mode': mode}, img_bgr)

    # Re-run the bin/dark masks separately so we can save them for diagnostics.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_mask = B.find_bin_mask(blurred, params)
    cfg = DETECTION_MODES[mode]
    _, dark = cv2.threshold(blurred, cfg['darkThreshold'], 255, cv2.THRESH_BINARY_INV)
    dark_in_bin = cv2.bitwise_and(dark, bin_mask)
    if cfg['morphClose']:
        k = int(params.get('close_kernel', 3))
        dark_in_bin = cv2.morphologyEx(dark_in_bin, cv2.MORPH_CLOSE,
                                       np.ones((k, k), np.uint8))

    # Production-equivalent counting pass.
    blobs, _ = B.detect_blobs(img_bgr, params)
    if sa_factor_override is not None:
        sa_factor = sa_factor_override
        sa_source = 'manual'
    else:
        sa_factor = B.predict_adaptive_factor(blobs, img_bgr.shape, mode)
        sa_source = 'adaptive'
    total, detections, single_area = B.blobs_to_count(blobs, sa_factor, params)

    # Overlay: bin contour in cyan, singles red, clumps orange with count + area.
    overlay = img_bgr.copy()
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), max(2, w // 600))
    for d in detections:
        if d['count'] == 1:
            cv2.circle(overlay, (d['cx'], d['cy']), max(7, w // 200),
                       (0, 0, 255), max(2, w // 600))
        else:
            cv2.circle(overlay, (d['cx'], d['cy']), max(11, w // 130),
                       (0, 140, 255), max(3, w // 500))
            cv2.putText(overlay, f"x{d['count']} ({d['area']})",
                        (d['cx'] + 14, d['cy'] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 2400),
                        (0, 140, 255), max(2, w // 700))

    header = f"pred={total}  truth={truth or '?'}  mode={mode}  " \
             f"sa={sa_factor:.2f}({sa_source})  sa_area={single_area:.0f}  " \
             f"blobs={len(blobs)}  bin={int((bin_mask > 0).sum()):,d}px"
    font_scale = max(0.9, w / 2200)
    cv2.putText(overlay, header, (20, int(40 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(3, w // 400))
    cv2.putText(overlay, header, (20, int(40 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(2, w // 600))

    cv2.imwrite(f"{out_prefix}_overlay.jpg", overlay)
    cv2.imwrite(f"{out_prefix}_binmask.jpg", bin_mask)
    cv2.imwrite(f"{out_prefix}_dark.jpg", dark_in_bin)

    # Per-blob diagnostic CSV — sort by area so you can scan the borderline
    # cases (right around the single-vs-clump cutoff) in one go.
    cutoff_lo = single_area * cfg.get('clumpRatioThreshold',
                                      params.get('clump_ratio_threshold', 1.6))
    cutoff_hi = single_area * cfg.get('largeClumpRatio',
                                      params.get('large_clump_ratio', 6.0))
    with open(f"{out_prefix}_blobs.csv", 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['area', 'ratio_to_single', 'count',
                     'cx', 'cy', 'reason'])
        for d in sorted(detections, key=lambda x: x['area']):
            ratio = d['area'] / single_area
            if ratio < cutoff_lo / single_area:
                reason = 'single'
            elif ratio < cutoff_hi / single_area:
                reason = f'small_clump(round={round(ratio)})'
            else:
                reason = f'large_clump(/{cfg.get("largeClumpOverlap", params.get("large_clump_overlap", 1.4))})'
            wr.writerow([d['area'], f'{ratio:.2f}', d['count'],
                         d['cx'], d['cy'], reason])

    areas = sorted(b['area'] for b in blobs)
    return {
        'truth': truth,
        'predicted': total,
        'diff': total - truth if truth else None,
        'sa_factor': round(sa_factor, 3),
        'single_area': round(single_area, 1),
        'num_blobs': len(blobs),
        'p10_area': areas[len(areas) // 10] if areas else None,
        'p50_area': areas[len(areas) // 2] if areas else None,
        'p90_area': areas[len(areas) * 9 // 10] if areas else None,
        'max_area': areas[-1] if areas else None,
        'cutoff_single_clump': round(cutoff_lo),
        'cutoff_large_clump': round(cutoff_hi),
        'n_singles': sum(1 for d in detections if d['count'] == 1),
        'n_clumps': sum(1 for d in detections if d['count'] > 1),
        'sum_clump_counts': sum(d['count'] for d in detections if d['count'] > 1),
    }


def _truth_from_name(name):
    """Filename pattern <id>_<count>.jpg yields the ground-truth count."""
    stem = Path(name).stem
    parts = stem.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+')
    parser.add_argument('--dir', default=str(TEST_FILES))
    parser.add_argument('--out-dir', default=str(ROOT / 'debug_out'))
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()),
                        default='sensitive')
    parser.add_argument('--sa-factor', type=float, default=None,
                        help='Override adaptive prediction with a fixed value.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{'name':<28} {'truth':>6} {'pred':>6} {'diff':>6} {'sa':>5} "
          f"{'sa_area':>8} {'blobs':>6} {'singles':>8} {'clumps':>7}")
    print('-' * 95)
    for name in args.images:
        path = os.path.join(args.dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"!! could not read {path}")
            continue
        out_prefix = os.path.join(args.out_dir, Path(name).stem)
        truth = _truth_from_name(name)
        info = visualize(img, args.detection, args.sa_factor, truth, out_prefix)
        diff_s = f'{info["diff"]:+d}' if info['diff'] is not None else '-'
        print(f"{name:<28} {info['truth'] or '?':>6} {info['predicted']:>6} "
              f"{diff_s:>6} {info['sa_factor']:>5.2f} "
              f"{info['single_area']:>8.0f} {info['num_blobs']:>6d} "
              f"{info['n_singles']:>8d} {info['n_clumps']:>7d}")


if __name__ == '__main__':
    main()
