"""
Joint sweep of clump/scaling hyperparameters on the combined 88-photo dataset.

Sweeps in two stages so we don't pay re-detection cost for cheap params:

  Stage 1 (post-detection, cheap): CLUMP_RATIO_THRESHOLD,
          LARGE_CLUMP_RATIO, LARGE_CLUMP_OVERLAP.
          Reuses cached blobs across the grid.

  Stage 2 (detection-changing, expensive): REFERENCE_PIXELS,
          linear-scale exponent. Re-detects per setting.

Runs with adaptive=off (fixed defaultSaFactor) so the signal isn't mixed
with the adaptive regression. Retrain adaptive once we pick a winner.

Usage:
  python scripts/tune_hyperparams.py
  python scripts/tune_hyperparams.py --detection sensitive --stage 1
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark
from benchmark import (
    DETECTION_MODES,
    TEST_FILES,
    blobs_to_count,
    detect_blobs,
    scale_params_for_image,
    CLUMP_RATIO_THRESHOLD,
    LARGE_CLUMP_RATIO,
    LARGE_CLUMP_OVERLAP,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_FILES2 = ROOT / 'test-files2'


def load_dataset(csv_path, image_dir):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('image_path'):
                rows.append((image_dir / r['image_path'], int(r['user_count'])))
    return rows


def detect_all(dataset, mode):
    cache = []
    base = {'detection_mode': mode}
    for path, truth in dataset:
        img = cv2.imread(str(path))
        if img is None:
            continue
        p = scale_params_for_image(base, img)
        blobs, _ = detect_blobs(img, p)
        cache.append({
            'path': str(path),
            'truth': truth,
            'blobs': blobs,
            'params': p,
            'shape': img.shape,
        })
    return cache


def evaluate(cache, sa_factor, overrides=None):
    overrides = overrides or {}
    errs_abs, errs_rel = [], []
    for c in cache:
        p = dict(c['params'])
        p.update(overrides)
        total, _, _ = blobs_to_count(c['blobs'], sa_factor, p)
        diff = abs(total - c['truth'])
        errs_abs.append(diff)
        errs_rel.append(diff / c['truth'] if c['truth'] > 0 else 0)
    return {
        'mae': float(np.mean(errs_abs)),
        'mape': float(np.mean(errs_rel) * 100),
    }


def stage1_counting_sweep(cache_all, cache_old, cache_new, mode, default_sa):
    """Sweep counting hyperparameters using cached blobs."""
    print(f'\n--- Stage 1: counting hyperparams ({mode}, sa={default_sa}, n={len(cache_all)}) ---')
    base = evaluate(cache_all, default_sa)
    print(f'Baseline: MAPE {base["mape"]:.3f}%  MAE {base["mae"]:.1f}\n')

    grid = {
        'clump_ratio_threshold': [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        'large_clump_ratio':     [4.0, 5.0, 6.0, 7.0, 8.0],
        'large_clump_overlap':   [1.2, 1.3, 1.4, 1.5, 1.6],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f'Grid: {len(combos)} combos')

    rows = []
    for vals in combos:
        ov = dict(zip(keys, vals))
        m = evaluate(cache_all, default_sa, ov)
        rows.append((ov, m))

    rows.sort(key=lambda r: r[1]['mape'])
    print(f'\nTop-10 (by combined MAPE):')
    print(f'  {"cr":>4} {"lcr":>5} {"lco":>5} {"MAPE":>7} {"MAE":>6} '
          f'{"old MAPE":>9} {"new MAPE":>9}')
    for ov, m in rows[:10]:
        m_old = evaluate(cache_old, default_sa, ov)
        m_new = evaluate(cache_new, default_sa, ov)
        flag = ''
        if (ov['clump_ratio_threshold'] == CLUMP_RATIO_THRESHOLD and
                ov['large_clump_ratio'] == LARGE_CLUMP_RATIO and
                ov['large_clump_overlap'] == LARGE_CLUMP_OVERLAP):
            flag = '  (= current)'
        print(f'  {ov["clump_ratio_threshold"]:>4.1f} {ov["large_clump_ratio"]:>5.1f} '
              f'{ov["large_clump_overlap"]:>5.2f} {m["mape"]:>6.2f}% {m["mae"]:>6.1f} '
              f'{m_old["mape"]:>8.2f}% {m_new["mape"]:>8.2f}%{flag}')

    print(f'\nDelta vs baseline: {rows[0][1]["mape"] - base["mape"]:+.3f}pp')
    return rows[0]  # (overrides, metrics) of winner


def stage2_detection_sweep(datasets_per_mode, mode, default_sa, stage1_winner):
    """Sweep REFERENCE_PIXELS and linear exponent. Re-detect per setting."""
    print(f'\n--- Stage 2: detection hyperparams ({mode}) ---')
    print(f'  Using stage-1 winning counting params: {stage1_winner[0]}')
    grid_ref = [800_000, 1_000_000, 1_250_000, 1_500_000, 2_000_000]
    grid_exp = [0.4, 0.5, 0.6]

    print(f'\nGrid: {len(grid_ref)} ref × {len(grid_exp)} exp = '
          f'{len(grid_ref) * len(grid_exp)} re-detections per dataset')
    print(f'  {"ref":>10} {"exp":>5} {"MAPE":>7} {"MAE":>6} '
          f'{"old MAPE":>9} {"new MAPE":>9}')

    results = []
    original_ref = benchmark.REFERENCE_PIXELS
    try:
        for ref_pixels in grid_ref:
            for exp in grid_exp:
                # Monkey-patch the scale function so we can vary both.
                benchmark.REFERENCE_PIXELS = ref_pixels
                orig_scale = benchmark.scale_params_for_image

                def custom_scale(base_params, img, _r=ref_pixels, _e=exp):
                    p = dict(base_params)
                    h, w = img.shape[:2]
                    area_scale = max(1.0, (h * w) / _r)
                    linear_scale = area_scale ** _e
                    mode_name = p.get('detection_mode', 'sensitive')
                    m = DETECTION_MODES[mode_name]
                    p.setdefault('area_scale', area_scale)
                    p.setdefault('linear_scale', linear_scale)
                    p.setdefault('min_blob_area',
                                 max(3, int(round(m['minBlobArea'] * area_scale))))
                    p.setdefault('min_single_area',
                                 benchmark.MIN_SINGLE_AREA * area_scale)
                    p.setdefault('density_blur',
                                 benchmark._odd(benchmark.DENSITY_BLUR * linear_scale))
                    p.setdefault('close_kernel', max(3, int(round(3 * linear_scale))))
                    p.setdefault('bin_dilate_kernel',
                                 max(3, int(round(20 * linear_scale))))
                    p.setdefault('bin_bright_close_kernel',
                                 max(3, int(round(25 * linear_scale))))
                    p.setdefault('bin_final_close_kernel',
                                 max(3, int(round(30 * linear_scale))))
                    return p

                benchmark.scale_params_for_image = custom_scale
                try:
                    cache_old = detect_all(datasets_per_mode['old'], mode)
                    cache_new = detect_all(datasets_per_mode['new'], mode)
                    cache_all = cache_old + cache_new
                    overrides = stage1_winner[0]
                    m_all = evaluate(cache_all, default_sa, overrides)
                    m_old = evaluate(cache_old, default_sa, overrides)
                    m_new = evaluate(cache_new, default_sa, overrides)
                    print(f'  {ref_pixels:>10,d} {exp:>5.2f} {m_all["mape"]:>6.2f}% '
                          f'{m_all["mae"]:>6.1f} {m_old["mape"]:>8.2f}% '
                          f'{m_new["mape"]:>8.2f}%')
                    results.append({
                        'ref_pixels': ref_pixels, 'exp': exp,
                        'mape': m_all['mape'], 'mape_old': m_old['mape'],
                        'mape_new': m_new['mape'],
                    })
                finally:
                    benchmark.scale_params_for_image = orig_scale
    finally:
        benchmark.REFERENCE_PIXELS = original_ref

    results.sort(key=lambda r: r['mape'])
    print(f'\nWinner: ref={results[0]["ref_pixels"]:,d}, exp={results[0]["exp"]} '
          f'-> MAPE {results[0]["mape"]:.3f}%')
    return results[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()))
    parser.add_argument('--stage', type=int, choices=[1, 2], default=None,
                        help='Run only stage 1 (counting) or 2 (detection). Default: both.')
    args = parser.parse_args()
    modes = [args.detection] if args.detection else list(DETECTION_MODES.keys())

    ds_old = load_dataset(TEST_FILES / 'feedback_rows.csv', TEST_FILES)
    ds_new = load_dataset(TEST_FILES2 / 'ground_truth.csv', TEST_FILES2)
    print(f'Loaded {len(ds_old)} test-files + {len(ds_new)} test-files2 = {len(ds_old)+len(ds_new)}')

    for mode in modes:
        print(f'\n{"#" * 60}\n### MODE: {mode}\n{"#" * 60}')
        default_sa = DETECTION_MODES[mode]['defaultSaFactor']

        cache_old = detect_all(ds_old, mode)
        cache_new = detect_all(ds_new, mode)
        cache_all = cache_old + cache_new

        stage1_winner = None
        if args.stage in (None, 1):
            stage1_winner = stage1_counting_sweep(cache_all, cache_old, cache_new,
                                                   mode, default_sa)

        if args.stage in (None, 2):
            if stage1_winner is None:
                # Use current production values
                stage1_winner = ({
                    'clump_ratio_threshold': CLUMP_RATIO_THRESHOLD,
                    'large_clump_ratio': LARGE_CLUMP_RATIO,
                    'large_clump_overlap': LARGE_CLUMP_OVERLAP,
                }, {})
            stage2_detection_sweep(
                {'old': ds_old, 'new': ds_new}, mode, default_sa, stage1_winner,
            )


if __name__ == '__main__':
    main()
