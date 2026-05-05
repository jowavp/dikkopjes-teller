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
- When `index.html` / `styles.css` / `app.js` change, bump `CACHE_VERSION`
  in `sw.js` so installed PWA clients pick up the new files.
- When user-visible behavior changes, bump `APP_VERSION` in `app.js`
  (shown in footer).

## Algorithm baseline

Lower-half median of blob areas, default `sa_factor = 0.97`. On the 46
ground-truth photos: MAPE 9.9%, 65% within 10%, bias +1.3. The previous
global log-histogram peak gave MAPE 11.4% with -11.6 undercounting bias.

The single-area estimator is the accuracy hot spot. Bin detection and
blob detection are mature; the remaining error is in converting blob
areas to counts via single-tadpole-area calibration.

## Why ~10% may be the unsupervised floor

`best_factor` per photo ranges 0.75-1.48. Linear regression on blob
features (total_dark, mean_area, …) explains only R² ≈ 0.23 of the
variance; with leave-one-out CV the best gain is MAPE 9.9% → 9.1%.
The residual variance is driven by things invisible in the blob
distribution: camera distance, lighting, water clarity, tadpole stage.

Paths beyond ~9% likely need (in order of leverage): a reference-size
object in the photo (gives true mm-per-pixel scale), per-user
calibration learned from the existing feedback stream, or a much
larger labelled dataset for a regression / tree-based model.
