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
test-files/                        ground-truth photos + feedback_rows.csv
```

## Run

```bash
python scripts/benchmark.py        # algorithm vs ground truth (46 photos)
python scripts/analyze_factor.py   # blob-feature → optimal-factor study
python scripts/debug_visualize.py <image.jpg>   # save pipeline overlays
```

Scripts resolve paths via `__file__` so cwd doesn't matter.

## Sync points

- `estimateSingleArea` in `app.js` and `estimate_single_area` in
  `scripts/benchmark.py` implement the same algorithm. The Python version
  is the benchmark source of truth — keep them aligned when changing.
- `DETECTION_MODES` in `app.js` and Python's `DETECTION_MODES` in
  `scripts/benchmark.py` must stay aligned (same threshold/min-area/close
  flags per mode).
- When `index.html` / `styles.css` / `app.js` change, bump `CACHE_VERSION`
  in `sw.js` so installed PWA clients pick up the new files.
- When user-visible behavior changes, bump `APP_VERSION` in `app.js`
  (shown in footer).

## Detection modes

The user picks between two detection modes (persisted in `localStorage`
under key `detectionMode`). Switching mode resets the slider to that mode's
default `sa_factor` and re-processes the current photo.

| mode      | dark thresh | morph close | min area | default sa_factor | MAPE | within-10% |
|-----------|-------------|-------------|----------|-------------------|------|------------|
| standard  | 75          | yes (3×3)   | 80       | 1.00              | 9.8% | 63%        |
| sensitive | 50          | no          | 30       | 1.20              | 8.5% | 78%        |

`sensitive` is the default. It wins on aggregate metrics by detecting more
frogs as separate blobs (so the area-counter has less work). Trade-off: on
photos where `standard` already nailed the count, `sensitive` typically
adds 5-15 extra counts (shadows or bright-but-dark non-frog regions slip
past the lower threshold).

Run benchmarks with `--detection {standard,sensitive}` to compare.

## Single-area estimator

Lower-half median of blob areas. Singles dominate the lower 50% of the
area distribution; clumps inflate the upper half, so the median of the
lower half tracks single-tadpole area robustly across clumping severity.
Replaces an earlier global log-histogram peak that was systematically
biased toward clumps.

## Why ~8% may be the unsupervised floor

We've explored several paths:
- **Watershed/opening clump splitting** (`scripts/diag_watershed.py`):
  failed because tadpoles in tight clumps merge into single connected
  regions in the binary mask — no internal structure for distance-
  transform to recover. Documented as `--algo watershed` in benchmark.py
  for reference.
- **Detection redesign** (`scripts/experiment_detection.py`): tested 25
  variants. `sensitive` mode (threshold 50, no close, smaller min-area)
  wins; adaptive/Otsu/percentile-based thresholds catastrophically fail.
- **Per-photo factor regression** (`scripts/analyze_factor.py`): R² ≈
  0.23, leave-one-out CV gives MAPE 9.9% → 9.1%. Limited by what blob
  features can predict.

Paths beyond ~8% likely need (in order of leverage): a reference-size
object in the photo (gives true mm-per-pixel scale), per-user
calibration learned from the existing feedback stream, or a labelled
dataset (~200+ photos) for a density-map CNN.
