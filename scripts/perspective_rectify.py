"""
Perspective rectification of the bin/tray. Finds the 4 corners of the
detected bin region and warps the image to a clean top-down view, so
that downstream blob detection runs on a perspective-corrected frame.

Used by tools/benchmarks; the warping itself is in `rectify_to_box`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import find_bin_mask, scale_params_for_image


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise from top-left: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # TR (small diff = x large relative to y)
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def _find_quad_brightness(img: np.ndarray):
    """Detect the box-bottom region by heavy-blur + brightness thresholding.

    Strategy: blur with a kernel ~1/10 of image dim, which destroys tadpole
    detail but preserves the large-scale brightness contrast between box
    interior (mid-bright water) and outside (darker surroundings or
    bright rim). Otsu on the blurred image partitions inside vs outside.
    The largest contour of the "inside" mask should match the box bottom.
    """
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aggressive blur so tadpoles dissolve into the surrounding water tone.
    blur_k = max(31, (min(w, h) // 10) | 1)  # odd
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Otsu partitions the blurred grayscale at its bimodal split.
    _, mid = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # We don't know a priori which side is "inside" — pick the one whose
    # largest connected component is biggest in the central 80% of the frame.
    central = np.zeros_like(mid)
    cv2.rectangle(central, (w // 10, h // 10), (w * 9 // 10, h * 9 // 10),
                  255, -1)
    cand_a = cv2.bitwise_and(mid, central)
    cand_b = cv2.bitwise_and(cv2.bitwise_not(mid), central)
    score_a = int(cv2.countNonZero(cand_a))
    score_b = int(cv2.countNonZero(cand_b))
    inside = mid if score_a >= score_b else cv2.bitwise_not(mid)

    # Close small gaps + remove specks.
    k = max(5, min(w, h) // 40)
    kernel = np.ones((k, k), np.uint8)
    inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, kernel)
    inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 'brightness_no_contour'
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < img_area * 0.20 or area > img_area * 0.98:
        return None, f'brightness_area_oob({area / img_area:.2f})'

    peri = cv2.arcLength(cnt, True)
    for eps_frac in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_points(approx.reshape(4, 2)), f'brightness@{eps_frac}'
    # Last resort: minAreaRect of the brightness region
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return _order_points(box), 'brightness_minAreaRect'


def _find_quad_mask(mask: np.ndarray):
    """Fallback: 4-corner quad from the bin mask convex hull."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 'no_contours'
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None, 'contour_too_small'
    peri = cv2.arcLength(cnt, True)
    for eps_frac in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(cnt, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_points(approx.reshape(4, 2)), f'mask_approx@{eps_frac}'
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return _order_points(box), 'minAreaRect_fallback'


def _find_quad(img: np.ndarray, mask: np.ndarray):
    """Try the brightness-based detector first; fall back to bin-mask hull."""
    quad, label = _find_quad_brightness(img)
    if quad is not None:
        return quad, label
    return _find_quad_mask(mask)


def rectify_to_box(img: np.ndarray, mode: str = 'sensitive', verbose: bool = False):
    """Warp `img` so the detected bin fills the output frame top-down.

    Returns (warped_img, info_dict) or (None, info_dict) if rectification fails.
    info_dict carries diagnostics: which corner-finding strategy worked, the
    source quad, and the output dimensions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    params = scale_params_for_image({'detection_mode': mode}, img)
    mask = find_bin_mask(blurred, params)

    quad, source = _find_quad(img, mask)
    info = {'source': source, 'quad': quad}
    if quad is None:
        return None, info

    # Output rectangle size: use the LONGER of opposite sides so we don't
    # squash a perspective-stretched edge.
    w = int(round(max(
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[3]),
    )))
    h = int(round(max(
        np.linalg.norm(quad[3] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
    )))
    if w < 100 or h < 100:
        info['source'] += '_rejected_small'
        return None, info
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    info['size'] = (w, h)
    if verbose:
        print(f'  rectify: {source}, out={w}x{h}, quad=\n{quad}')
    return warped, info


if __name__ == '__main__':
    """Visual sanity check: rectify a handful of photos and save side-by-side."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='debug_out/rectify')
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.paths:
        img = cv2.imread(p)
        if img is None:
            print(f'Skip (cannot read): {p}')
            continue
        warped, info = rectify_to_box(img, verbose=True)
        name = Path(p).stem
        if warped is None:
            print(f'  FAILED: {p} ({info["source"]})')
            continue
        # Draw quad on a copy of the original
        overlay = img.copy()
        pts = info['quad'].astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), max(3, img.shape[0] // 300))
        cv2.imwrite(str(out_dir / f'{name}_quad.jpg'), overlay)
        cv2.imwrite(str(out_dir / f'{name}_warped.jpg'), warped)
        print(f'  saved: {out_dir}/{name}_quad.jpg + _warped.jpg')
