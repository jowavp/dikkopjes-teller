"""
Side-by-side comparison: baseline vs tight50 detection on each photo.
Shows where tight50 wins big, ties, or regresses.
"""
import csv
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import find_bin_mask, blobs_to_count, TEST_FILES
from experiment_detection import detect_baseline, detect_tight_threshold_50


def main():
    rows = []
    with open(TEST_FILES / 'feedback_rows.csv', 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append(r)

    sa_b = 1.00
    sa_c = 1.20

    results = []
    for r in rows:
        img = cv2.imread(str(TEST_FILES / r['image_path']))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        bin_mask = find_bin_mask(blurred)
        bb = detect_baseline(img, blurred, bin_mask)
        cb = detect_tight_threshold_50(img, blurred, bin_mask)
        truth = int(r['user_count'])
        tb, _, _ = blobs_to_count(bb, sa_b)
        tc, _, _ = blobs_to_count(cb, sa_c)
        results.append({
            'image': r['image_path'][:30],
            'truth': truth,
            'base': tb, 'base_err': tb - truth, 'base_abs': abs(tb - truth),
            'tight': tc, 'tight_err': tc - truth, 'tight_abs': abs(tc - truth),
            'delta': abs(tb - truth) - abs(tc - truth),  # positive = tight better
        })

    results.sort(key=lambda x: -x['delta'])  # biggest wins for tight first

    print(f"{'image':<32} {'truth':>5} {'base':>5} {'tight':>5} "
          f"{'base_err':>9} {'tight_err':>10} {'delta':>6}")
    print('-' * 80)
    for r in results:
        print(f"{r['image']:<32} {r['truth']:>5} {r['base']:>5} {r['tight']:>5} "
              f"{r['base_err']:>+9} {r['tight_err']:>+10} {r['delta']:>+6}")
    print('-' * 80)

    wins = sum(1 for r in results if r['delta'] > 0)
    ties = sum(1 for r in results if r['delta'] == 0)
    losses = sum(1 for r in results if r['delta'] < 0)
    big_wins = sum(1 for r in results if r['delta'] > 10)
    big_losses = sum(1 for r in results if r['delta'] < -10)
    print(f"tight50 vs baseline: {wins} wins, {ties} ties, {losses} losses")
    print(f"  big wins (>10):    {big_wins}")
    print(f"  big losses (<-10): {big_losses}")

    # Per-bucket: was baseline already good? did tight50 wreck it?
    base_was_good = [r for r in results if r['base_abs'] <= 10]
    base_was_bad = [r for r in results if r['base_abs'] > 30]
    print(f"\nWhen baseline was good (abs_err <= 10, n={len(base_was_good)}):")
    print(f"  tight50 average abs_err: {np.mean([r['tight_abs'] for r in base_was_good]):.1f}")
    print(f"\nWhen baseline was bad (abs_err > 30, n={len(base_was_bad)}):")
    print(f"  tight50 average abs_err: {np.mean([r['tight_abs'] for r in base_was_bad]):.1f}")
    print(f"  baseline average abs_err: {np.mean([r['base_abs'] for r in base_was_bad]):.1f}")


if __name__ == '__main__':
    main()
