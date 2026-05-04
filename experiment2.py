"""Sweep more parameters to find a robust improvement."""
import csv
import os
import statistics
import numpy as np
import cv2
import benchmark as B
from experiment import (
    estimate_single_area_median,
    estimate_single_area_trimmed,
    estimate_single_area_log_kde,
    blobs_to_count_with_estimator,
    evaluate,
)


def main():
    rows = []
    with open('test-files/feedback_rows.csv', 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    print(f"Evaluating on {len(rows)} test images")
    print('-' * 110)

    print("\n=== Sweep MIN_BLOB_AREA with current estimator (histogram peak), sa=0.95 ===")
    for mba in [30, 50, 60, 70, 80, 90, 100, 120, 150]:
        evaluate(rows, 'test-files', B.estimate_single_area, 0.95,
                 f"current, MIN_BLOB={mba:3d}, sa=0.95", {'min_blob_area': mba})

    print("\n=== Sweep MIN_BLOB_AREA with log-space estimator, sa=0.95 ===")
    for mba in [30, 50, 60, 70, 80, 90, 100, 120, 150]:
        evaluate(rows, 'test-files', estimate_single_area_log_kde, 0.95,
                 f"log-peak, MIN_BLOB={mba:3d}, sa=0.95", {'min_blob_area': mba})

    print("\n=== Best MIN_BLOB sweep sa-factor with log-space ===")
    p = {'min_blob_area': 80}
    for sa in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
        evaluate(rows, 'test-files', estimate_single_area_log_kde, sa,
                 f"log-peak, MIN_BLOB=80, sa={sa:.2f}", p)

    print("\n=== Best MIN_BLOB sweep sa-factor with current ===")
    p = {'min_blob_area': 80}
    for sa in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
        evaluate(rows, 'test-files', B.estimate_single_area, sa,
                 f"current, MIN_BLOB=80, sa={sa:.2f}", p)

    print("\n=== Sweep DARK_THRESHOLD with MIN_BLOB=80, current estimator ===")
    for dt in [60, 70, 75, 80, 85, 90, 95]:
        p = {'min_blob_area': 80, 'dark_threshold': dt}
        evaluate(rows, 'test-files', B.estimate_single_area, 0.95,
                 f"current, DARK={dt:3d}, MIN_BLOB=80", p)


if __name__ == '__main__':
    main()
