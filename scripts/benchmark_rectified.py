"""
Benchmark the dikkopjes-teller algorithm on perspective-rectified images.

Each input image is first warped to a top-down view of its box, then the
standard detect_blobs + adaptive sa_factor + blobs_to_count pipeline runs
on the warped image.

Usage:
  python scripts/benchmark_rectified.py
  python scripts/benchmark_rectified.py --detection standard
  python scripts/benchmark_rectified.py --no-rectify   # control: original images
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
    predict_adaptive_factor,
    scale_params_for_image,
)
from perspective_rectify import rectify_to_box

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES2 = ROOT / 'test-files2'


def evaluate(dataset_label, csv_path, image_dir, detection_mode, rectify):
    abs_errs, rel_errs = [], []
    skipped_rectify = 0
    rows = []
    with open(csv_path, 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    for r in rows:
        path = image_dir / r['image_path']
        img = cv2.imread(str(path))
        if img is None:
            continue
        truth = int(r['user_count'])

        used_img = img
        rect_info = None
        if rectify:
            warped, rect_info = rectify_to_box(img, mode=detection_mode)
            if warped is None:
                skipped_rectify += 1
            else:
                used_img = warped

        scaled = scale_params_for_image({'detection_mode': detection_mode}, used_img)
        blobs, _ = detect_blobs(used_img, scaled)
        sa_factor = predict_adaptive_factor(blobs, used_img.shape, detection_mode)
        total, _, _ = blobs_to_count(blobs, sa_factor, scaled)
        diff = abs(total - truth)
        abs_errs.append(diff)
        rel_errs.append(diff / truth if truth > 0 else 0)

    n = len(abs_errs)
    mape = float(np.mean(rel_errs)) * 100
    mae = float(np.mean(abs_errs))
    within10 = sum(1 for e in rel_errs if e <= 0.10) / n * 100
    print(f'  {dataset_label}: MAPE {mape:.2f}%  MAE {mae:.1f}  '
          f'within-10% {within10:.0f}%  (rectify-skipped: {skipped_rectify}/{n})')
    return mape, mae, within10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()),
                        default='sensitive')
    parser.add_argument('--no-rectify', action='store_true',
                        help='Control: run on original images (should reproduce production)')
    args = parser.parse_args()
    rectify = not args.no_rectify
    print(f'### detection={args.detection}  rectify={rectify}\n')
    evaluate('test-files', TEST_FILES / 'feedback_rows.csv',
             TEST_FILES, args.detection, rectify)
    evaluate('test-files2', TEST_FILES2 / 'ground_truth.csv',
             TEST_FILES2, args.detection, rectify)


if __name__ == '__main__':
    main()
