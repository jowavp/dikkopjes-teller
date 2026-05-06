"""
Diagnostic: voor een paar foto's, toon per blob hoeveel cores watershed vindt
en hoeveel area-based zou voorspellen. Helpt begrijpen waarom watershed faalt.
"""
import csv
import os
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import (
    detect_blobs_full, estimate_single_area,
    count_blob_watershed, count_blob_opening,
    TEST_FILES, ROOT,
    CLUMP_RATIO_THRESHOLD,
)


def main():
    # Pick a few photos with different clumping severity
    targets = [
        '1777904960538-yjrtpgf8.jpg',  # truth 369, heavy clumping
        '1777905120016-moet9qe4.jpg',  # truth 327, mid clumping
        '1777904747650-yyxmeski.jpg',  # truth 309, moderate
    ]
    truth_map = {r['image_path']: int(r['user_count'])
                 for r in csv.DictReader(open(TEST_FILES / 'feedback_rows.csv'))
                 if r.get('image_path')}

    for img_name in targets:
        img = cv2.imread(str(TEST_FILES / img_name))
        truth = truth_map.get(img_name)
        blobs, _, labels = detect_blobs_full(img)
        single_area = estimate_single_area([b['area'] for b in blobs], 0.97)
        radius = float(np.sqrt(single_area / np.pi))

        # Per-blob breakdown — only show clumps
        clumps = [b for b in blobs if b['area'] / single_area > CLUMP_RATIO_THRESHOLD]
        print(f"\n=== {img_name} (truth={truth}, n_blobs={len(blobs)}, "
              f"n_clumps={len(clumps)}, single_area={single_area:.0f}, R={radius:.1f}px) ===")

        sum_area_count = 0
        sum_ws_count = 0
        sum_open_count = 0
        for b in sorted(clumps, key=lambda x: -x['area'])[:8]:
            ratio = b['area'] / single_area
            area_count = round(ratio)
            ws_count = count_blob_watershed(b, labels, single_area,
                                            {'peak_thresh_ratio': 0.45})
            open_count = count_blob_opening(b, labels, single_area, {'open_ratio': 0.6})
            sum_area_count += area_count
            sum_ws_count += ws_count
            sum_open_count += open_count
            print(f"  blob area={b['area']:>5}  ratio={ratio:5.2f}  "
                  f"area-est={area_count}  watershed={ws_count}  opening={open_count}")
        print(f"  ... (showing top 8 clumps)")
        print(f"  sum (top 8): area={sum_area_count}, ws={sum_ws_count}, open={sum_open_count}")

        # Save the largest clump as a diagnostic image
        biggest = max(clumps, key=lambda x: x['area'])
        x, y, w, h = biggest['bbox']
        region = (labels[y:y + h, x:x + w] == biggest['label']).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(region, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        # Normalize for viz
        dist_viz = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 else dist.astype(np.uint8)
        # Cores at threshold 0.45*R
        _, cores = cv2.threshold(dist, 0.45 * radius, 255, cv2.THRESH_BINARY)
        cores = cores.astype(np.uint8)

        out_dir = ROOT / 'debug_out'
        out_dir.mkdir(exist_ok=True)
        stem = Path(img_name).stem
        # Stack horizontally: mask | distance | cores
        h_max = max(padded.shape[0], dist_viz.shape[0], cores.shape[0])
        def pad(m, h_target):
            if m.shape[0] < h_target:
                m = cv2.copyMakeBorder(m, 0, h_target - m.shape[0], 0, 0,
                                       cv2.BORDER_CONSTANT, value=0)
            return m
        strip = np.hstack([
            pad(padded, h_max), pad(dist_viz, h_max), pad(cores, h_max),
        ])
        # Scale up if small
        if strip.shape[1] < 600:
            scale = 600 // max(1, strip.shape[1]) + 1
            strip = cv2.resize(strip, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f'{stem}_biggest_clump.png'), strip)


if __name__ == '__main__':
    main()
