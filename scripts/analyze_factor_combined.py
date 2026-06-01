"""
Per-photo adaptive sa_factor regression on the COMBINED dataset (test-files
+ test-files2), with image megapixels added as a feature.

Outputs:
- Pearson correlations of blob features with best_factor
- In-sample and leave-one-out CV MAPE for top-k feature subsets
- Recommended regression coefficients (mean / std / beta) for shipping

Compared to scripts/analyze_factor.py:
- Loads both ground-truth sets, not just test-files/
- Adds image_megapixels feature (the per-folder optima diverge, so resolution
  is almost certainly predictive)
- Baseline = current production default per mode (0.98 std / 1.21 sens)
- Clip range widened to [0.70, 1.40] (test-files2 sometimes wants > 1.25)
- Trains per mode separately (the two modes need different coefficients)

Usage:
  python scripts/analyze_factor_combined.py
  python scripts/analyze_factor_combined.py --detection sensitive
  python scripts/analyze_factor_combined.py --emit-coefs  # print JSON for shipping
"""
import argparse
import csv
import json
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

# Clip range for predicted factor — widened beyond analyze_factor.py's 0.70..1.25
# because high-res photos sometimes legitimately want up to 1.30.
FACTOR_MIN = 0.70
FACTOR_MAX = 1.40


def features_from_blobs(blobs, img_shape):
    """Per-photo features computable at inference time (no truth)."""
    if not blobs:
        return None
    areas = np.array([b['area'] for b in blobs], dtype=np.float64)
    n = len(areas)
    sorted_a = np.sort(areas)
    median = float(np.median(areas))
    half = sorted_a[: max(3, n // 2)]
    lh_median = float(np.median(half))
    clump_frac = float(np.mean(areas > 1.5 * lh_median))
    largest = float(sorted_a[-1])
    p90 = float(sorted_a[int(0.9 * (n - 1))])
    h, w = img_shape[:2]
    mp = (h * w) / 1_000_000.0
    return {
        'num_blobs': n,
        'median_area': median,
        'lh_median': lh_median,
        'mean_area': float(areas.mean()),
        'std_area': float(areas.std()),
        'cv_area': float(areas.std() / areas.mean()) if areas.mean() > 0 else 0.0,
        'total_dark': float(areas.sum()),
        'clump_frac': clump_frac,
        'p90_over_median': p90 / median if median > 0 else 0.0,
        'largest_over_median': largest / median if median > 0 else 0.0,
        'log_density': float(np.log(n / max(1.0, areas.sum()))) if areas.sum() > 0 else 0.0,
        'megapixels': mp,
        'log_megapixels': float(np.log(max(mp, 0.01))),
    }


def best_factor_for(blobs, target, params):
    best = None
    for i in range(101):
        f = 0.50 + i * 0.01
        total, _, _ = blobs_to_count(blobs, f, params)
        d = abs(total - target)
        if best is None or d < best[1]:
            best = (round(f, 2), d, total)
    return best


def load_dataset(csv_path, image_dir, detection_mode):
    """Detect blobs once per photo; return cache list of dicts."""
    cache = []
    with open(csv_path, 'r', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r.get('image_path')]
    base_params = {'detection_mode': detection_mode}
    for r in rows:
        path = image_dir / r['image_path']
        img = cv2.imread(str(path))
        if img is None:
            continue
        scaled = scale_params_for_image(base_params, img)
        blobs, _ = detect_blobs(img, scaled)
        if not blobs:
            continue
        feats = features_from_blobs(blobs, img.shape)
        truth = int(r['user_count'])
        bf, _, _ = best_factor_for(blobs, truth, scaled)
        cache.append({
            'path': str(path),
            'truth': truth,
            'blobs': blobs,
            'params': scaled,
            'feats': feats,
            'best_factor': bf,
        })
    return cache


def metrics(errs_abs, errs_rel):
    return {
        'mae': float(np.mean(errs_abs)),
        'mape': float(np.mean(errs_rel) * 100),
    }


def fit_linreg(X, y):
    """Standardize X, fit OLS, return (beta, mu, sd)."""
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1
    Xn = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(Xn, y, rcond=None)
    return beta, mu, sd


def predict(beta, mu, sd, x):
    xn = np.concatenate([[1.0], (x - mu) / sd])
    return float(xn @ beta)


def evaluate_factor_strategy(cache, factor_fn):
    errs_abs, errs_rel = [], []
    for c in cache:
        f = factor_fn(c)
        f = max(FACTOR_MIN, min(FACTOR_MAX, f))
        total, _, _ = blobs_to_count(c['blobs'], f, c['params'])
        diff = abs(total - c['truth'])
        errs_abs.append(diff)
        errs_rel.append(diff / c['truth'] if c['truth'] > 0 else 0)
    return metrics(errs_abs, errs_rel)


def analyze_mode(mode, emit_coefs=False):
    print(f'\n{"#" * 60}')
    print(f'### MODE: {mode}')
    print(f'{"#" * 60}')
    default = DETECTION_MODES[mode]['defaultSaFactor']

    cache_old = load_dataset(TEST_FILES / 'feedback_rows.csv', TEST_FILES, mode)
    cache_new = load_dataset(TEST_FILES2 / 'ground_truth.csv', TEST_FILES2, mode)
    cache = cache_old + cache_new
    n = len(cache)
    print(f'\nLoaded {len(cache_old)} test-files + {len(cache_new)} test-files2 = {n} photos')

    feat_names = list(cache[0]['feats'].keys())
    y = np.array([c['best_factor'] for c in cache])
    print(f'best_factor: mean={y.mean():.3f}, std={y.std():.3f}, '
          f'range=[{y.min():.2f}, {y.max():.2f}]')

    # Correlations
    print('\nPearson correlation with best_factor (sorted by |r|):')
    print(f'{"feature":<22} {"r":>7}')
    corrs = []
    for f in feat_names:
        vals = np.array([c['feats'][f] for c in cache])
        if vals.std() == 0:
            r = 0.0
        else:
            r = float(np.corrcoef(vals, y)[0, 1])
        corrs.append((f, r))
    for f, r in sorted(corrs, key=lambda x: -abs(x[1])):
        print(f'  {f:<22} {r:>+.3f}')

    # In-sample fit on all features
    X_all = np.array([[c['feats'][f] for f in feat_names] for c in cache])
    beta_all, mu_all, sd_all = fit_linreg(X_all, y)
    pred = X_all @ ((beta_all[1:]) / sd_all) + (beta_all[0] - (beta_all[1:] * mu_all / sd_all).sum())
    r2 = 1 - ((y - pred).var() / y.var())
    print(f'\nIn-sample OLS (all {len(feat_names)} features): R² = {r2:.3f}')

    # Baseline: fixed default
    base = evaluate_factor_strategy(cache, lambda c: default)
    print(f'\nBaseline (sa_factor={default}): MAE {base["mae"]:.1f}, MAPE {base["mape"]:.2f}%')

    # In-sample (optimistic) on top-k feature subsets
    top_sorted = [f for f, _ in sorted(corrs, key=lambda x: -abs(x[1]))]
    print('\nIn-sample (overschat optimisme) per feature subset:')
    print(f'  {"subset":<8} {"MAE":>6} {"MAPE":>7}')
    for k in (1, 2, 3, 5, len(feat_names)):
        subset = top_sorted[:k]
        X = np.array([[c['feats'][f] for f in subset] for c in cache])
        beta, mu, sd = fit_linreg(X, y)
        m = evaluate_factor_strategy(
            cache,
            lambda c, s=subset, b=beta, mu=mu, sd=sd:
                predict(b, mu, sd, np.array([c['feats'][f] for f in s])),
        )
        print(f'  top{k:<4}    {m["mae"]:>6.1f} {m["mape"]:>6.2f}%')

    # LOOCV (honest generalization)
    print('\nLeave-one-out CV (eerlijke generalisatie):')
    print(f'  {"subset":<8} {"MAE":>6} {"MAPE":>7} {"vs base":>8}')
    loocv_results = {}
    for k in (1, 2, 3, 5, len(feat_names)):
        subset = top_sorted[:k]
        preds = []
        for hold in range(n):
            train_idx = [i for i in range(n) if i != hold]
            X_tr = np.array([[cache[i]['feats'][f] for f in subset] for i in train_idx])
            y_tr = y[train_idx]
            beta, mu, sd = fit_linreg(X_tr, y_tr)
            x_te = np.array([cache[hold]['feats'][f] for f in subset])
            preds.append(predict(beta, mu, sd, x_te))

        def factor_fn(c, _preds=preds, _idx=[0]):
            v = _preds[_idx[0]]
            _idx[0] += 1
            return v

        m = evaluate_factor_strategy(cache, factor_fn)
        delta = m['mape'] - base['mape']
        flag = '  WORSE' if delta > 0 else ''
        print(f'  top{k:<4}    {m["mae"]:>6.1f} {m["mape"]:>6.2f}% {delta:>+7.2f}%{flag}')
        loocv_results[k] = (subset, m)

    # Per-folder breakdown of best LOOCV strategy
    print('\nLOOCV per-folder breakdown:')
    print(f'  {"subset":<8} {"old MAPE":>10} {"new MAPE":>10}')
    for k in (3, 5, len(feat_names)):
        subset = top_sorted[:k]
        old_errs, new_errs = [], []
        for hold in range(n):
            train_idx = [i for i in range(n) if i != hold]
            X_tr = np.array([[cache[i]['feats'][f] for f in subset] for i in train_idx])
            y_tr = y[train_idx]
            beta, mu, sd = fit_linreg(X_tr, y_tr)
            c = cache[hold]
            x_te = np.array([c['feats'][f] for f in subset])
            pred_f = max(FACTOR_MIN, min(FACTOR_MAX, predict(beta, mu, sd, x_te)))
            total, _, _ = blobs_to_count(c['blobs'], pred_f, c['params'])
            err = abs(total - c['truth']) / c['truth'] if c['truth'] > 0 else 0
            if hold < len(cache_old):
                old_errs.append(err)
            else:
                new_errs.append(err)
        print(f'  top{k:<4}    {np.mean(old_errs)*100:>9.2f}% {np.mean(new_errs)*100:>9.2f}%')

    # Emit shipping coefficients. We pin k=5: it captures most of the LOOCV
    # win (-1.11% standard, -2.08% sensitive) with much lower overfitting risk
    # than the full 13-feature model on n=88. Also halves the JS code surface.
    if emit_coefs:
        best_k = 5
        subset, _ = loocv_results[best_k]
        X = np.array([[c['feats'][f] for f in subset] for c in cache])
        beta, mu, sd = fit_linreg(X, y)
        coefs = {
            'mode': mode,
            'features': subset,
            'beta': beta.tolist(),
            'mu': mu.tolist(),
            'sd': sd.tolist(),
            'factor_min': FACTOR_MIN,
            'factor_max': FACTOR_MAX,
            'baseline_factor': default,
        }
        print(f'\n--- shipping coefs (top{best_k}) ---')
        print(json.dumps(coefs, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection', choices=list(DETECTION_MODES.keys()))
    parser.add_argument('--emit-coefs', action='store_true')
    args = parser.parse_args()
    modes = [args.detection] if args.detection else list(DETECTION_MODES.keys())
    for m in modes:
        analyze_mode(m, emit_coefs=args.emit_coefs)


if __name__ == '__main__':
    main()
