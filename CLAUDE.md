# Dikkopjes-teller

Pure-browser PWA that counts tadpoles in photos via OpenCV.js (classical
image processing, no AI / no backend inference). An optional feedback flow
ships corrected counts and blob features to Supabase for offline tuning.

## Layout

```
index.html / styles.css / app.js   web app
sw.js, manifest.json, *.png        PWA assets
stats.html                         standalone stats viewer
scripts/                           Python (benchmark, analysis, debug viz)
benchmark_results/                 CSV output of benchmark.py
test-files/                        ≈1 MP ground-truth photos + feedback_rows.csv
test-files2/                       ≈6 MP ground-truth photos + ground_truth.csv
```

## Run

```bash
python scripts/benchmark.py                       # 46-photo (1 MP) set
python scripts/benchmark.py --dir test-files2 \   # 42-photo (6 MP) set
  --csv test-files2/ground_truth.csv
python scripts/tune_sa_factor.py                  # sweep defaultSaFactor
python scripts/analyze_factor_combined.py         # train adaptive regression
python scripts/debug_visualize.py <image.jpg>     # save pipeline overlays
```

Scripts resolve paths via `__file__` so cwd doesn't matter.

## Sync points

- `estimateSingleArea` in `app.js` and `estimate_single_area` in
  `scripts/benchmark.py` implement the same algorithm. The Python version
  is the benchmark source of truth — keep them aligned when changing.
- `DETECTION_MODES` in `app.js` and Python's `DETECTION_MODES` in
  `scripts/benchmark.py` must stay aligned (same threshold/min-area/close
  flags + `defaultSaFactor`).
- `ADAPTIVE_MODELS` (regression coefficients for adaptive sa_factor) in
  `app.js` and `scripts/benchmark.py` must stay aligned. Retrain with
  `python scripts/analyze_factor_combined.py --emit-coefs` and paste the
  JSON into both files. `computeBlobFeatures` (JS) /
  `compute_blob_features` (Py) / `features_from_blobs` (analyze script)
  must compute the same features in the same order.
- Resolution scaling: `REFERENCE_PIXELS` + `computeResolutionScale`
  (JS) / `scale_params_for_image` (Py) keep absolute-pixel constants
  (min_blob_area, MIN_SINGLE_AREA, density blur, morph kernels) calibrated
  to image megapixels. Clamps to 1.0 minimum so calibration-set behavior
  is preserved.
- When `index.html` / `styles.css` / `app.js` change, bump `CACHE_VERSION`
  in `sw.js` so installed PWA clients pick up the new files.
- When user-visible behavior changes, bump `APP_VERSION` in `app.js`
  (shown in footer).

## Detection modes

The user picks between two detection modes (persisted in `localStorage`
under key `detectionMode`). On photo upload or mode change the adaptive
factor regression (see below) picks a starting `sa_factor`; the user can
override via the slider + Process button.

Per-mode MAPE / within-10% with adaptive factor on the combined 88-photo
benchmark (in-sample; LOOCV — honest generalization — is ~0.5pp worse):

| mode      | dark thresh | morph close | min area | default sa_factor | MAPE (1 MP / 6 MP) | within-10% (1 MP / 6 MP) |
|-----------|-------------|-------------|----------|-------------------|--------------------|--------------------------|
| standard  | 75          | yes (3×3)   | 80       | 0.98              | 8.7% / 8.7%        | 65% / 60%                |
| sensitive | 50          | no          | 30       | 1.21              | 7.4% / 7.0%        | 70% / 81%                |

`sensitive` is the default. It wins on aggregate metrics by detecting more
frogs as separate blobs (so the area-counter has less work). Trade-off: on
photos where `standard` already nailed the count, `sensitive` typically
adds 5-15 extra counts (shadows or bright-but-dark non-frog regions slip
past the lower threshold).

Run benchmarks with `--detection {standard,sensitive}` to compare.
`--adaptive off` disables the regression and falls back to `defaultSaFactor`.

## Single-area estimator

Lower-half median of blob areas. Singles dominate the lower 50% of the
area distribution; clumps inflate the upper half, so the median of the
lower half tracks single-tadpole area robustly across clumping severity.
Replaces an earlier global log-histogram peak that was systematically
biased toward clumps.

## Adaptive sa_factor regression

`ADAPTIVE_MODELS` is a small per-mode linear regression that predicts a
photo-specific `sa_factor` from blob features, replacing the per-mode
constant default. Trained by `scripts/analyze_factor_combined.py` on the
combined 88-photo dataset; top-5 features per mode (clump_frac dominates
for sensitive, r=+0.60).

LOOCV improvement vs the fixed-default baseline:
- standard:  10.4% → **9.3%** MAPE  (-1.1pp)
- sensitive:  9.8% → **7.7%** MAPE  (-2.1pp)

Predictions are clipped to [0.70, 1.40]. The Process button forces the
slider value (manual override) instead of adaptive.

## Why ~7% may be the unsupervised floor

We've explored several paths:
- **Watershed/opening clump splitting** (`scripts/diag_watershed.py`):
  failed because tadpoles in tight clumps merge into single connected
  regions in the binary mask — no internal structure for distance-
  transform to recover. Documented as `--algo watershed` in benchmark.py
  for reference.
- **Detection redesign** (`scripts/experiment_detection.py`): tested 25
  variants. `sensitive` mode (threshold 50, no close, smaller min-area)
  wins; adaptive/Otsu/percentile-based thresholds catastrophically fail.
- **Per-photo factor regression** (`scripts/analyze_factor_combined.py`):
  see above — got us from ~10% to ~7-9% LOOCV.

Paths beyond ~7% likely need (in order of leverage): a reference-size
object in the photo (gives true mm-per-pixel scale), per-user
calibration learned from the existing feedback stream, or a labelled
dataset (~200+ photos) for a density-map CNN.
