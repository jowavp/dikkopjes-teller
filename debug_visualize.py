"""
Debug: visualize the algorithm pipeline for problem images.
Saves intermediate masks (bin mask, dark mask, blobs) so we can see what's going wrong.
"""
import os
import argparse
import numpy as np
import cv2
import benchmark as B


def visualize(img_bgr, sa_factor, out_prefix):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_mask = B.find_bin_mask(blurred)

    _, dark = cv2.threshold(blurred, B.DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    dark_in_bin = cv2.bitwise_and(dark, bin_mask)
    dark_in_bin = cv2.morphologyEx(dark_in_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    blobs, _ = B.detect_blobs(img_bgr)
    total, detections, single_area = B.blobs_to_count(blobs, sa_factor)

    # Compose visualization image
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    # Draw bin mask boundary in cyan
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 4)

    for d in detections:
        if d['count'] == 1:
            cv2.circle(overlay, (d['cx'], d['cy']), 7, (0, 0, 255), 2)
        else:
            cv2.circle(overlay, (d['cx'], d['cy']), 11, (0, 140, 255), 3)
            cv2.putText(overlay, f"x{d['count']} ({d['area']})",
                        (d['cx'] + 13, d['cy'] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    cv2.putText(overlay, f"pred={total} sa={sa_factor:.2f} sa_area={single_area:.0f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
    cv2.putText(overlay, f"pred={total} sa={sa_factor:.2f} sa_area={single_area:.0f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    # Save: overlay, bin mask, dark mask
    cv2.imwrite(f"{out_prefix}_overlay.jpg", overlay)
    cv2.imwrite(f"{out_prefix}_binmask.jpg", bin_mask)
    cv2.imwrite(f"{out_prefix}_dark.jpg", dark_in_bin)

    # Histogram of blob areas
    areas = sorted(b['area'] for b in blobs)
    return {
        'total': total,
        'num_blobs': len(blobs),
        'single_area': single_area,
        'p10': areas[len(areas) // 10] if areas else None,
        'p50': areas[len(areas) // 2] if areas else None,
        'p90': areas[len(areas) * 9 // 10] if areas else None,
        'max': areas[-1] if areas else None,
        'biggest_clump_count': max((d['count'] for d in detections), default=0),
        'sum_clump_2plus_count': sum(d['count'] for d in detections if d['count'] > 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+', help='image filenames (in test-files/ by default)')
    parser.add_argument('--dir', default='test-files')
    parser.add_argument('--out-dir', default='debug_out')
    parser.add_argument('--sa-factor', type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.images:
        path = os.path.join(args.dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"!! could not read {path}")
            continue
        out_prefix = os.path.join(args.out_dir, os.path.splitext(name)[0])
        info = visualize(img, args.sa_factor, out_prefix)
        print(f"{name:<40} {info}")


if __name__ == '__main__':
    main()
