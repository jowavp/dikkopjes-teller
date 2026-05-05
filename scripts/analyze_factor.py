"""
Onderzoekt of de optimale sa_factor te voorspellen is uit blob-features.
Doel: zo ja, kunnen we 'm dynamisch zetten zonder gebruikersinput.
"""
import csv
import os
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import detect_blobs, estimate_single_area, blobs_to_count, statistics_median, TEST_FILES


def features_from_blobs(blobs):
    """Per-photo features that are computable at inference time (no truth)."""
    if not blobs:
        return None
    areas = np.array([b['area'] for b in blobs], dtype=np.float64)
    n = len(areas)
    sorted_a = np.sort(areas)
    median = float(np.median(areas))
    half = sorted_a[: max(3, n // 2)]
    lh_median = float(np.median(half))  # = current single area estimate
    # Hoeveel blobs lijken clumps (>1.5x lower-half median)?
    clump_frac = float(np.mean(areas > 1.5 * lh_median))
    largest = float(sorted_a[-1])
    p90 = float(sorted_a[int(0.9 * (n - 1))])
    return {
        'num_blobs': n,
        'median_area': median,
        'lh_median': lh_median,
        'mean_area': float(areas.mean()),
        'std_area': float(areas.std()),
        'cv_area': float(areas.std() / areas.mean()) if areas.mean() > 0 else 0,
        'total_dark': float(areas.sum()),
        'clump_frac': clump_frac,
        'p90_over_median': p90 / median if median > 0 else 0,
        'largest_over_median': largest / median if median > 0 else 0,
        'log_density': np.log(n / max(1.0, areas.sum())) if areas.sum() > 0 else 0,
    }


def best_factor_for(blobs, target):
    best = None
    for i in range(101):
        f = 0.50 + i * 0.01  # 0.50..1.50
        total, _, _ = blobs_to_count(blobs, f)
        d = abs(total - target)
        if best is None or d < best[1]:
            best = (round(f, 2), d, total)
    return best


def main():
    rows = list(csv.DictReader(open(TEST_FILES / 'feedback_rows.csv')))
    rows = [r for r in rows if r.get('image_path')]
    data = []
    for r in rows:
        path = os.path.join(TEST_FILES, r['image_path'])
        img = cv2.imread(path)
        if img is None:
            continue
        blobs, _ = detect_blobs(img)
        if not blobs:
            continue
        truth = int(r['user_count'])
        feats = features_from_blobs(blobs)
        f, d, _ = best_factor_for(blobs, truth)
        feats['best_factor'] = f
        feats['truth'] = truth
        data.append(feats)

    n = len(data)
    print(f'n photos: {n}')

    # Correlations: each feature vs best_factor (Pearson)
    feat_names = [k for k in data[0].keys() if k not in ('best_factor', 'truth')]
    bf = np.array([d['best_factor'] for d in data])
    print(f'\nbest_factor stats: mean={bf.mean():.3f}, std={bf.std():.3f}, '
          f'min={bf.min():.2f}, max={bf.max():.2f}')

    print('\nPearson correlation with best_factor:')
    print(f'{"feature":<22} {"r":>6}  {"|r|":>5}')
    correlations = []
    for f in feat_names:
        vals = np.array([d[f] for d in data])
        if vals.std() == 0:
            r = 0
        else:
            r = float(np.corrcoef(vals, bf)[0, 1])
        correlations.append((f, r))
    for f, r in sorted(correlations, key=lambda x: -abs(x[1])):
        print(f'{f:<22} {r:>+.3f}  {abs(r):.3f}')

    # Simple linear regression with top features
    top = [f for f, r in sorted(correlations, key=lambda x: -abs(x[1]))[:3]]
    print(f'\nLinear regression on top 3: {top}')
    X = np.array([[d[f] for f in top] for d in data])
    # standardize
    Xm, Xs = X.mean(0), X.std(0)
    Xs[Xs == 0] = 1
    Xn = (X - Xm) / Xs
    # add bias
    Xn = np.column_stack([np.ones(n), Xn])
    coefs, *_ = np.linalg.lstsq(Xn, bf, rcond=None)
    pred_bf = Xn @ coefs
    residuals = bf - pred_bf
    rmse_bf = np.sqrt((residuals ** 2).mean())
    print(f'  RMSE on best_factor: {rmse_bf:.3f} (vs std-only baseline {bf.std():.3f})')
    print(f'  R²: {1 - (residuals.var() / bf.var()):.3f}')

    # Cache blobs per row to avoid re-running OpenCV
    cache = []
    for r in rows:
        path = os.path.join(TEST_FILES, r['image_path'])
        img = cv2.imread(path)
        if img is None:
            continue
        blobs, _ = detect_blobs(img)
        if not blobs:
            continue
        feats = features_from_blobs(blobs)
        cache.append({
            'blobs': blobs,
            'truth': int(r['user_count']),
            'feats': feats,
        })

    print('\n=== In-sample (overschat optimisme) ===')
    abs_errors_predicted, rel_predicted = [], []
    abs_errors_default, rel_default = [], []
    for c in cache:
        x = np.array([c['feats'][f] for f in top])
        xn = (x - Xm) / Xs
        xn = np.concatenate([[1], xn])
        pred_f = float(xn @ coefs)
        pred_f = max(0.70, min(1.25, pred_f))
        total_default, _, _ = blobs_to_count(c['blobs'], 0.97)
        total_pred, _, _ = blobs_to_count(c['blobs'], pred_f)
        truth = c['truth']
        abs_errors_default.append(abs(total_default - truth))
        abs_errors_predicted.append(abs(total_pred - truth))
        rel_default.append(abs(total_default - truth) / truth)
        rel_predicted.append(abs(total_pred - truth) / truth)
    print(f'  Default sa=0.97: MAE {np.mean(abs_errors_default):.1f}, MAPE {np.mean(rel_default)*100:.1f}%')
    print(f'  Predicted (in-sample): MAE {np.mean(abs_errors_predicted):.1f}, MAPE {np.mean(rel_predicted)*100:.1f}%')

    # Leave-one-out cross-validation (eerlijke schatting)
    print('\n=== Leave-one-out CV (eerlijke generalisatie) ===')
    feat_subsets = [
        ('top1', top[:1]),
        ('top2', top[:2]),
        ('top3', top[:3]),
        ('all',  feat_names),
    ]
    for name, subset in feat_subsets:
        errs_pred, errs_default = [], []
        rel_pred = []
        for hold_out in range(n):
            train_idx = [i for i in range(n) if i != hold_out]
            X_tr = np.array([[cache[i]['feats'][f] for f in subset] for i in train_idx])
            y_tr = np.array([bf[i] for i in train_idx])
            mu, sd = X_tr.mean(0), X_tr.std(0)
            sd[sd == 0] = 1
            X_tr_n = np.column_stack([np.ones(len(train_idx)), (X_tr - mu) / sd])
            beta, *_ = np.linalg.lstsq(X_tr_n, y_tr, rcond=None)

            x_te = np.array([cache[hold_out]['feats'][f] for f in subset])
            x_te_n = np.concatenate([[1], (x_te - mu) / sd])
            pred_f = float(x_te_n @ beta)
            pred_f = max(0.70, min(1.25, pred_f))
            total_pred, _, _ = blobs_to_count(cache[hold_out]['blobs'], pred_f)
            truth = cache[hold_out]['truth']
            errs_pred.append(abs(total_pred - truth))
            rel_pred.append(abs(total_pred - truth) / truth)
        print(f'  {name:>5} ({len(subset)} feats): MAE {np.mean(errs_pred):.1f}, '
              f'MAPE {np.mean(rel_pred)*100:.1f}%')

    print(f'\nTheoretical floor (per-photo optimal): MAPE ~ 1-2%')


if __name__ == '__main__':
    main()
