'use strict';

const APP_VERSION = '0.7.1-tuned-defaults';  // bump on releases
document.getElementById('appVersion').textContent = 'v' + APP_VERSION;

// Threshold used for bin/tray detection (finding WHERE the bak is in the photo).
// Independent of the tadpole detection mode below — tweaking this would affect
// which region we constrain counting to, not how individual frogs are detected.
const BIN_DARK_THRESHOLD = 75;
const BIN_BRIGHT_THRESHOLD = 95;
const DENSITY_BLUR = 201;
const DENSITY_THRESHOLD = 6;
const MIN_SINGLE_AREA = 150.0;
const CLUMP_RATIO_THRESHOLD = 1.6;
const LARGE_CLUMP_RATIO = 6.0;
const LARGE_CLUMP_OVERLAP = 1.4;

// Resolution-aware scaling. The pixel constants above were calibrated on
// photos spanning 0.76–1.23 MP (test-files/). Higher-res photos need them
// scaled — areas linearly with pixel count, kernel/blur sizes with sqrt.
// Reference is the top of the calibration range so all calibration photos
// clamp to scale=1.0 (original behavior preserved exactly).
const REFERENCE_PIXELS = 1_250_000;

function computeResolutionScale(width, height) {
  const areaScale = Math.max(1.0, (width * height) / REFERENCE_PIXELS);
  return { areaScale, linearScale: Math.sqrt(areaScale) };
}

function oddKernel(n) {
  const r = Math.max(1, Math.round(n));
  return r % 2 === 1 ? r : r + 1;
}

/**
 * Tadpole detection modes. Choose how aggressively we binarize the in-bin
 * grayscale into "tadpole" vs "background" pixels.
 *
 * Benchmark (46 ground-truth photos):
 *   standard  : MAPE 9.8%, 63% within 10% of truth, bias -7.5
 *   sensitive : MAPE 8.5%, 78% within 10% of truth, bias -0.9
 *
 * sensitive wins on the within-10% metric (more consistent for the user) but
 * regresses on photos where standard already nailed it (~5-15 frog overcount
 * because shadows/dim non-frog regions creep in past the lower threshold).
 * standard is a safer default for clean, well-lit photos.
 */
const DETECTION_MODES = {
  standard: {
    label: 'Standaard',
    darkThreshold: 75,
    morphClose: true,
    minBlobArea: 80,
    // Empirical optimum across the combined 88-photo dataset (test-files +
    // test-files2). See scripts/tune_sa_factor.py.
    defaultSaFactor: 0.98,
  },
  sensitive: {
    label: 'Gevoelig',
    darkThreshold: 50,
    morphClose: false,
    minBlobArea: 30,
    // Empirical optimum across the combined 88-photo dataset.
    defaultSaFactor: 1.21,
  },
};
const DEFAULT_DETECTION_MODE = 'sensitive';

function getDetectionMode() {
  const m = localStorage.getItem('detectionMode');
  return (m && DETECTION_MODES[m]) ? m : DEFAULT_DETECTION_MODE;
}

let cvReady = false;
let currentImage = null;
let lastResult = null;  // {total, saFactor, image, blobs, stats} - gevuld door processImage()

const $ = id => document.getElementById(id);
const status = $('status');
const statusCard = $('statusCard');
const uploadZone = $('uploadZone');
const fileInput = $('fileInput');
const controls = $('controls');
const processBtn = $('processBtn');
const saFactorInput = $('saFactor');
const saFactorValue = $('saFactorValue');
const detectionModeSelect = $('detectionMode');
const resultCard = $('resultCard');
const resultCount = $('resultCount');
const resultCanvas = $('resultCanvas');
const downloadBtn = $('downloadBtn');

function onOpenCvReady() {
  // OpenCV.js needs a beat after load
  cv['onRuntimeInitialized'] = () => {
    cvReady = true;
    window.cvReady = true;  // for tests
    window.countTadpoles = countTadpoles;  // for tests
    status.innerHTML = '✓ Klaar — laad een foto op';
    status.classList.remove('error');
  };
}

function onOpenCvError() {
  status.innerHTML = '⚠️ OpenCV.js kon niet ingeladen worden. Controleer je internetverbinding en herlaad de pagina.';
  status.classList.add('error');
}

// Laad OpenCV.js dynamisch zodat de handlers gegarandeerd gedefinieerd zijn
// vóór het script element ge-add wordt (vermijdt race met cached opencv.js).
(function loadOpenCv() {
  const s = document.createElement('script');
  s.async = true;
  s.src = 'https://docs.opencv.org/4.10.0/opencv.js';
  s.onload = onOpenCvReady;
  s.onerror = onOpenCvError;
  document.head.appendChild(s);
})();

// --- Upload handling ---
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) loadImage(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
  if (e.target.files.length) loadImage(e.target.files[0]);
});

function loadImage(file) {
  if (!file.type.startsWith('image/')) {
    showError('Dat is geen geldige foto.');
    return;
  }
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      currentImage = img;
      controls.classList.remove('hidden');
      uploadZone.querySelector('p strong').textContent = '✓ ' + file.name;
      status.innerHTML = 'Klaar om te tellen';
      // Auto-trigger if cv is ready
      if (cvReady) processImage();
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

saFactorInput.addEventListener('input', () => {
  saFactorValue.textContent = parseFloat(saFactorInput.value).toFixed(2);
});

processBtn.addEventListener('click', processImage);

// --- Detection mode (persists in localStorage) ---
function applyDetectionMode(mode, opts = {}) {
  if (!DETECTION_MODES[mode]) mode = DEFAULT_DETECTION_MODE;
  localStorage.setItem('detectionMode', mode);
  if (detectionModeSelect) detectionModeSelect.value = mode;
  const cfg = DETECTION_MODES[mode];
  // Reset slider to this mode's optimum unless we're explicitly preserving
  if (!opts.preserveSlider) {
    saFactorInput.value = cfg.defaultSaFactor.toFixed(2);
    saFactorValue.textContent = cfg.defaultSaFactor.toFixed(2);
  }
}
applyDetectionMode(getDetectionMode());

detectionModeSelect?.addEventListener('change', () => {
  applyDetectionMode(detectionModeSelect.value);
  if (currentImage && cvReady) processImage();
});

function showError(msg) {
  status.textContent = msg;
  status.classList.add('error');
  statusCard.classList.remove('hidden');
}

function setStatus(msg, withSpinner = false) {
  status.classList.remove('error');
  status.innerHTML = (withSpinner ? '<span class="spinner"></span>' : '') + msg;
  statusCard.classList.remove('hidden');
}

// --- Image processing ---
async function processImage() {
  if (!cvReady) { showError('OpenCV.js is nog niet klaar, even geduld.'); return; }
  if (!currentImage) { showError('Eerst een foto opladen.'); return; }

  processBtn.disabled = true;
  setStatus('Foto wordt geanalyseerd…', true);
  resultCard.classList.add('hidden');
  resetFeedbackUi();

  // Allow the UI to update before the heavy work starts
  await new Promise(r => setTimeout(r, 50));

  try {
    const saFactor = parseFloat(saFactorInput.value);
    // Detect blobs once (heavy), then convert to detections (cheap)
    const { blobs, srcRgb, minSingleArea } = detectBlobs(currentImage);
    const { detections, total } = blobsToDetections(blobs, saFactor, minSingleArea);
    // Bereken samenvattende stats voor analytics
    const stats = computeBlobStats(blobs, detections, saFactor, minSingleArea);
    // Render with annotations
    const canvas = renderResult(srcRgb, detections, total);
    const dest = resultCanvas;
    dest.width = canvas.width;
    dest.height = canvas.height;
    dest.getContext('2d').drawImage(canvas, 0, 0);
    resultCount.textContent = `🐸 ${total} dikkopjes geteld`;
    // Bewaar laatste resultaat (incl. blobs voor snelle hercalibratie + stats voor analytics)
    lastResult = {
      total, saFactor, image: currentImage, blobs, stats, minSingleArea,
      imageWidth: currentImage.naturalWidth,
      imageHeight: currentImage.naturalHeight,
    };
    resultCard.classList.remove('hidden');
    statusCard.classList.add('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error(err);
    showError('Fout tijdens analyse: ' + err.message);
  } finally {
    processBtn.disabled = false;
  }
}

downloadBtn.addEventListener('click', () => {
  resultCanvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dikkopjes_geteld.jpg';
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/jpeg', 0.92);
});

// --- The actual tadpole counting (port of the Python script) ---

/**
 * Doet de zware OpenCV stappen (bin mask + blob detection) en geeft de
 * blob-lijst en hun area-array terug. Ook de bron-Mat (RGB out) en gray.
 * Caller moet src/grey/etc niet vrijgeven (dit doen we hier).
 *
 * Geeft {blobs, src_rgb}, waarbij blobs = [{cx, cy, area}].
 */
function detectBlobs(img) {
  const cfg = DETECTION_MODES[getDetectionMode()];
  const { areaScale, linearScale } = computeResolutionScale(
    img.naturalWidth, img.naturalHeight
  );
  const minBlobArea = Math.max(3, Math.round(cfg.minBlobArea * areaScale));
  const closeK = Math.max(3, Math.round(3 * linearScale));
  const minSingleArea = MIN_SINGLE_AREA * areaScale;

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = img.naturalWidth;
  srcCanvas.height = img.naturalHeight;
  srcCanvas.getContext('2d').drawImage(img, 0, 0);

  const src = cv.imread(srcCanvas);  // RGBA
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  const binMask = findBinMask(blurred, linearScale);

  const dark = new cv.Mat();
  cv.threshold(blurred, dark, cfg.darkThreshold, 255, cv.THRESH_BINARY_INV);
  cv.bitwise_and(dark, binMask, dark);
  if (cfg.morphClose) {
    const closeKernel = cv.Mat.ones(closeK, closeK, cv.CV_8U);
    cv.morphologyEx(dark, dark, cv.MORPH_CLOSE, closeKernel);
    closeKernel.delete();
  }

  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const numLabels = cv.connectedComponentsWithStats(dark, labels, stats, centroids, 8);

  const blobs = [];
  for (let i = 1; i < numLabels; i++) {
    const area = stats.intAt(i, cv.CC_STAT_AREA);
    if (area < minBlobArea) continue;
    const cx = Math.round(centroids.doubleAt(i, 0));
    const cy = Math.round(centroids.doubleAt(i, 1));
    blobs.push({ area, cx, cy });
  }

  // Strip alpha for drawing
  const srcRgb = new cv.Mat();
  cv.cvtColor(src, srcRgb, cv.COLOR_RGBA2RGB);

  src.delete(); gray.delete(); blurred.delete(); binMask.delete();
  dark.delete(); labels.delete(); stats.delete(); centroids.delete();
  return { blobs, srcRgb, minSingleArea };
}

/**
 * Vertaal blobs naar tellingen + detections aan de hand van saFactor (puur JS, geen OpenCV).
 */
function blobsToDetections(blobs, saFactor, minSingleArea = MIN_SINGLE_AREA) {
  if (blobs.length === 0) return { detections: [], total: 0 };
  const singleArea = estimateSingleArea(blobs.map(b => b.area), saFactor, minSingleArea);
  const detections = [];
  let total = 0;
  for (const b of blobs) {
    const ratio = b.area / singleArea;
    let count;
    if (ratio < CLUMP_RATIO_THRESHOLD) count = 1;
    else if (ratio < LARGE_CLUMP_RATIO) count = Math.round(ratio);
    else count = Math.round(b.area / (singleArea * LARGE_CLUMP_OVERLAP));
    detections.push({ cx: b.cx, cy: b.cy, count });
    total += count;
  }
  return { detections, total };
}

/**
 * Berekent samenvattende statistieken over een set blobs + detecties.
 * Wordt meegestuurd naar Supabase voor analytics + om later een adaptief model te trainen.
 *
 *   num_singles      : detecties met count == 1
 *   num_small_clumps : detecties met count 2..5
 *   num_medium_clumps: detecties met count 6..15
 *   num_large_clumps : detecties met count 16+
 */
function computeBlobStats(blobs, detections, saFactor, minSingleArea = MIN_SINGLE_AREA) {
  const stats = {
    num_blobs: blobs.length,
    num_singles: 0,
    num_small_clumps: 0,
    num_medium_clumps: 0,
    num_large_clumps: 0,
    single_area_estimate: null,
    total_dark_area: 0,
    largest_blob_area: 0,
    median_blob_area: null,
  };
  if (blobs.length === 0) return stats;

  stats.single_area_estimate = Math.round(
    estimateSingleArea(blobs.map(b => b.area), saFactor, minSingleArea)
  );
  for (const b of blobs) {
    stats.total_dark_area += b.area;
    if (b.area > stats.largest_blob_area) stats.largest_blob_area = b.area;
  }
  const sorted = blobs.map(b => b.area).sort((a, b) => a - b);
  stats.median_blob_area = sorted[Math.floor(sorted.length / 2)];

  for (const d of detections) {
    if (d.count === 1) stats.num_singles++;
    else if (d.count <= 5) stats.num_small_clumps++;
    else if (d.count <= 15) stats.num_medium_clumps++;
    else stats.num_large_clumps++;
  }
  return stats;
}

/**
 * Tekent de annotaties op de bron-Mat en geeft een canvas terug.
 * Consumeert (deletes) srcRgb.
 */
function renderResult(srcRgb, detections, total) {
  const out = srcRgb;
  const RED = new cv.Scalar(255, 0, 0);
  const ORANGE = new cv.Scalar(255, 140, 0);
  for (const d of detections) {
    if (d.count === 1) {
      cv.circle(out, new cv.Point(d.cx, d.cy), 7, RED, 2, cv.LINE_AA);
    } else {
      cv.circle(out, new cv.Point(d.cx, d.cy), 11, ORANGE, 3, cv.LINE_AA);
      cv.putText(out, 'x' + d.count, new cv.Point(d.cx + 13, d.cy + 6),
                 cv.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2, cv.LINE_AA);
    }
  }

  const banner = `Aantal dikkopjes: ${total}`;
  cv.rectangle(out, new cv.Point(10, 10), new cv.Point(760, 90),
               new cv.Scalar(255, 255, 255), -1);
  cv.rectangle(out, new cv.Point(10, 10), new cv.Point(760, 90),
               new cv.Scalar(0, 0, 0), 2);
  cv.putText(out, banner, new cv.Point(25, 65),
             cv.FONT_HERSHEY_SIMPLEX, 1.8, new cv.Scalar(0, 0, 0), 4, cv.LINE_AA);

  const outRgba = new cv.Mat();
  cv.cvtColor(out, outRgba, cv.COLOR_RGB2RGBA);
  const resultCv = document.createElement('canvas');
  cv.imshow(resultCv, outRgba);

  out.delete(); outRgba.delete();
  return resultCv;
}

function countTadpoles(img, saFactor) {
  const { blobs, srcRgb, minSingleArea } = detectBlobs(img);
  const { detections, total } = blobsToDetections(blobs, saFactor, minSingleArea);
  const canvas = renderResult(srcRgb, detections, total);
  return { canvas, total };
}

function findBinMask(blurred, linearScale = 1.0) {
  const densityBlur = oddKernel(DENSITY_BLUR * linearScale);
  const kDilate = Math.max(3, Math.round(20 * linearScale));
  const kBright = Math.max(3, Math.round(25 * linearScale));
  const kFinal = Math.max(3, Math.round(30 * linearScale));

  // 1. Dark pixels — uses BIN_DARK_THRESHOLD (independent of tadpole detection mode)
  const dark = new cv.Mat();
  cv.threshold(blurred, dark, BIN_DARK_THRESHOLD, 255, cv.THRESH_BINARY_INV);

  // 2. Heavy blur = density map
  const density = new cv.Mat();
  cv.GaussianBlur(dark, density, new cv.Size(densityBlur, densityBlur), 0);
  const dense = new cv.Mat();
  cv.threshold(density, dense, DENSITY_THRESHOLD, 255, cv.THRESH_BINARY);

  // 3. Take ALL dense components >= 20% of the largest. Critical for bins
  // where tadpoles cluster in disconnected areas (e.g. left vs right side):
  // taking only the largest component would crop out half the bin.
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const num = cv.connectedComponentsWithStats(dense, labels, stats, centroids, 8);
  let largestArea = 0;
  for (let i = 1; i < num; i++) {
    const a = stats.intAt(i, cv.CC_STAT_AREA);
    if (a > largestArea) largestArea = a;
  }
  const keepThreshold = Math.max(1, Math.floor(largestArea * 0.20));
  const keep = new Uint8Array(num);
  for (let i = 1; i < num; i++) {
    if (stats.intAt(i, cv.CC_STAT_AREA) >= keepThreshold) keep[i] = 1;
  }

  let mask;
  if (largestArea === 0) {
    mask = new cv.Mat(blurred.rows, blurred.cols, cv.CV_8U, new cv.Scalar(255));
  } else {
    const region = new cv.Mat(labels.rows, labels.cols, cv.CV_8U, new cv.Scalar(0));
    const labelsData = labels.data32S;  // Int32Array view
    const regionData = region.data;     // Uint8Array view
    for (let i = 0; i < labelsData.length; i++) {
      if (keep[labelsData[i]]) regionData[i] = 255;
    }
    // Convex hull from contours
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(region, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    if (contours.size() === 0) {
      mask = new cv.Mat(blurred.rows, blurred.cols, cv.CV_8U, new cv.Scalar(255));
    } else {
      // Combine all contour points into one Mat for the convex hull
      // (matches the Python `np.vstack(contours)` then `cv.convexHull(...)`)
      const pts = [];
      for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i);
        for (let j = 0; j < c.rows; j++) {
          pts.push(c.intAt(j, 0), c.intAt(j, 1));
        }
      }
      const ptsMat = cv.matFromArray(pts.length / 2, 1, cv.CV_32SC2, pts);
      const hull = new cv.Mat();
      cv.convexHull(ptsMat, hull);

      mask = cv.Mat.zeros(blurred.rows, blurred.cols, cv.CV_8U);
      const hullVec = new cv.MatVector();
      hullVec.push_back(hull);
      cv.drawContours(mask, hullVec, -1, new cv.Scalar(255), -1);

      // Dilate slightly
      const kDilateMat = cv.Mat.ones(kDilate, kDilate, cv.CV_8U);
      cv.dilate(mask, mask, kDilateMat);
      kDilateMat.delete();

      // Combine with bright zones
      const bright = new cv.Mat();
      cv.threshold(blurred, bright, BIN_BRIGHT_THRESHOLD, 255, cv.THRESH_BINARY);
      const kBrightMat = cv.Mat.ones(kBright, kBright, cv.CV_8U);
      // Python had iterations=3; OpenCV.js's iteration arg is unreliable across builds, so loop:
      for (let i = 0; i < 3; i++) {
        cv.morphologyEx(bright, bright, cv.MORPH_CLOSE, kBrightMat);
      }
      cv.bitwise_and(mask, bright, mask);
      bright.delete();
      kBrightMat.delete();

      // Close gaps
      const kFinalMat = cv.Mat.ones(kFinal, kFinal, cv.CV_8U);
      cv.morphologyEx(mask, mask, cv.MORPH_CLOSE, kFinalMat);
      kFinalMat.delete();

      ptsMat.delete();
      hull.delete();
      hullVec.delete();
    }
    region.delete();
    contours.delete();
    hierarchy.delete();
  }

  dark.delete(); density.delete(); dense.delete();
  labels.delete(); stats.delete(); centroids.delete();
  return mask;
}

function estimateSingleArea(areas, saFactor, minSingleArea = MIN_SINGLE_AREA) {
  // Lower-half median: take the median of the smallest 50% of blob areas.
  // Singles dominate the lower half of the area distribution; clumps inflate
  // the upper half. The previous log-histogram peak was systematically biased
  // toward the clump cluster on photos with heavy clumping (>50% of blobs),
  // causing undercounting. Lower-half median tracks the singles cluster
  // robustly across clumping severity.
  //
  // Benchmark (46 ground-truth photos): MAPE 11.4% → 9.8%; 46% → 65% within
  // 10% of truth; bias -11.6 → +1.3.
  if (areas.length === 0) return minSingleArea * saFactor;
  if (areas.length < 3) {
    const sorted = [...areas].sort((a, b) => a - b);
    return Math.max(minSingleArea, sorted[Math.floor(sorted.length / 2)]) * saFactor;
  }
  const sorted = [...areas].sort((a, b) => a - b);
  const halfLen = Math.max(3, Math.floor(sorted.length / 2));
  const half = sorted.slice(0, halfLen);
  const median = half[Math.floor(half.length / 2)];
  return Math.max(minSingleArea, median) * saFactor;
}

// --- PWA: service worker + install prompt ---

// 1. Service worker registreren (zorgt dat de app installeerbaar is en offline werkt)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(err => {
      console.warn('Service worker registratie mislukt:', err);
    });
  });
}

// 2. Android/Chrome: vang het beforeinstallprompt event op en toon eigen install knop
let deferredPrompt = null;
const installBanner = document.getElementById('installBanner');
const installBtn = document.getElementById('installBtn');
const dismissInstallBtn = document.getElementById('dismissInstallBtn');

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  if (!localStorage.getItem('installDismissed') && !isStandalone()) {
    installBanner.classList.remove('hidden');
  }
});

installBtn?.addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  deferredPrompt = null;
  installBanner.classList.add('hidden');
  if (outcome === 'dismissed') {
    localStorage.setItem('installDismissed', '1');
  }
});

dismissInstallBtn?.addEventListener('click', () => {
  installBanner.classList.add('hidden');
  localStorage.setItem('installDismissed', '1');
});

window.addEventListener('appinstalled', () => {
  installBanner.classList.add('hidden');
  deferredPrompt = null;
});

// 3. iOS Safari: geen beforeinstallprompt event, dus zelf detecteren en hint tonen
function isIosSafari() {
  const ua = navigator.userAgent;
  const isIos = /iPad|iPhone|iPod/.test(ua) ||
                (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const isSafari = /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS/.test(ua);
  return isIos && isSafari;
}
function isStandalone() {
  return window.matchMedia('(display-mode: standalone)').matches ||
         window.navigator.standalone === true;
}
if (isIosSafari() && !isStandalone() && !localStorage.getItem('iosHintDismissed')) {
  document.getElementById('iosInstallHint').classList.remove('hidden');
}
document.getElementById('dismissIosBtn')?.addEventListener('click', () => {
  document.getElementById('iosInstallHint').classList.add('hidden');
  localStorage.setItem('iosHintDismissed', '1');
});

// --- Feedback systeem (Supabase) ---

const SUPABASE_URL = 'https://fraaipwqqqechmeyvsvb.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZyYWFpcHdxcXFlY2htZXl2c3ZiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc4ODk0MjQsImV4cCI6MjA5MzQ2NTQyNH0.mY7e9E8KJY-009BED0fldx_dkkoP2ScW7WlG6D4EGuA';

const feedbackPrompt = $('feedbackPrompt');
const feedbackForm = $('feedbackForm');
const feedbackThanks = $('feedbackThanks');
const feedbackThanksDetail = $('feedbackThanksDetail');
const userCountInput = $('userCountInput');
const autoCalibSuggestion = $('autoCalibSuggestion');
const calibText = $('calibText');
const applyCalibBtn = $('applyCalibBtn');

function resetFeedbackUi() {
  feedbackPrompt?.classList.remove('hidden');
  feedbackForm?.classList.add('hidden');
  feedbackThanks?.classList.add('hidden');
  autoCalibSuggestion?.classList.add('hidden');
  if (userCountInput) userCountInput.value = '';
  // Reset share buttons in case ze nog disabled waren van vorige feedback
  const sb = $('shareBtn'), nsb = $('noShareBtn');
  if (sb) { sb.disabled = false; sb.textContent = '📤 Ja, deel'; }
  if (nsb) { nsb.disabled = false; }
}

// "Klopt": stuur enkel de basics op (geen foto), bedank en sluit
$('feedbackYesBtn')?.addEventListener('click', async () => {
  if (!lastResult) return;
  feedbackPrompt.classList.add('hidden');
  feedbackThanks.classList.remove('hidden');
  feedbackThanksDetail.textContent = '';
  // Async, niet wachten. Bij bevestiging IS saFactor effectief de beste factor.
  sendFeedback({
    appCount: lastResult.total,
    userCount: lastResult.total,  // bevestigd
    saFactor: lastResult.saFactor,
    bestSaFactor: lastResult.saFactor,  // huidige factor was correct
    sharePhoto: false,
    notes: 'confirmed_correct',
  }).catch(err => console.warn('Feedback verzenden mislukt:', err));
});

// "Niet klopt": toon invoerveld
$('feedbackNoBtn')?.addEventListener('click', () => {
  feedbackPrompt.classList.add('hidden');
  feedbackForm.classList.remove('hidden');
  setTimeout(() => userCountInput.focus(), 100);
});

// Bij intypen: bereken auto-calibratie en toon suggestie
let calibTimer = null;
userCountInput?.addEventListener('input', () => {
  clearTimeout(calibTimer);
  calibTimer = setTimeout(() => {
    const userN = parseInt(userCountInput.value, 10);
    if (!Number.isFinite(userN) || userN < 1 || !lastResult) {
      autoCalibSuggestion.classList.add('hidden');
      return;
    }
    const best = findBestSaFactor(lastResult.image, userN);
    if (!best || Math.abs(best.predictedTotal - userN) > 5) {
      autoCalibSuggestion.classList.add('hidden');
      return;
    }
    if (Math.abs(best.factor - lastResult.saFactor) < 0.005) {
      autoCalibSuggestion.classList.add('hidden');
      return;
    }
    calibText.textContent =
      `Met factor ${best.factor.toFixed(2)} zou ik ${best.predictedTotal} tellen.`;
    applyCalibBtn.dataset.factor = best.factor.toString();
    autoCalibSuggestion.classList.remove('hidden');
  }, 350);
});

// Toepassen: pas slider aan, herbereken telling
applyCalibBtn?.addEventListener('click', () => {
  const factor = parseFloat(applyCalibBtn.dataset.factor);
  if (!Number.isFinite(factor)) return;
  saFactorInput.value = factor.toFixed(2);
  saFactorValue.textContent = factor.toFixed(2);
  autoCalibSuggestion.classList.add('hidden');
  // Herbereken
  processImage();
});

// "Ja, deel" / "Liever niet": stuur feedback (met of zonder foto) en bedank
$('shareBtn')?.addEventListener('click', () => submitFeedback(true));
$('noShareBtn')?.addEventListener('click', () => submitFeedback(false));

async function submitFeedback(sharePhoto) {
  const userN = parseInt(userCountInput.value, 10);
  if (!Number.isFinite(userN) || userN < 0) {
    userCountInput.focus();
    userCountInput.style.borderColor = '#c0392b';
    return;
  }
  const shareBtn = $('shareBtn');
  const noShareBtn = $('noShareBtn');
  shareBtn.disabled = true; noShareBtn.disabled = true;
  shareBtn.textContent = '⏳ Versturen…';

  try {
    // findBestSaFactor is synchroon en bijna gratis (gebruikt gecachte blobs)
    const best = findBestSaFactor(lastResult.image, userN);
    await sendFeedback({
      appCount: lastResult.total,
      userCount: userN,
      saFactor: lastResult.saFactor,
      bestSaFactor: best?.factor,
      sharePhoto: sharePhoto,
      image: sharePhoto ? lastResult.image : null,
    });
    feedbackForm.classList.add('hidden');
    feedbackThanks.classList.remove('hidden');
    feedbackThanksDetail.textContent =
      sharePhoto
        ? 'Je foto + telling helpen mee om de app te verbeteren.'
        : 'Je telling is anoniem geregistreerd.';
  } catch (err) {
    console.error('submitFeedback error:', err);
    shareBtn.disabled = false; noShareBtn.disabled = false;
    shareBtn.textContent = '📤 Ja, deel';
    alert('Versturen mislukt: ' + (err.message || err) + '\n\nProbeer het later opnieuw.');
  }
}

/**
 * Vindt de sa_factor die het dichtst bij userTarget komt.
 * Gebruikt de gecachte blobs van lastResult — geen OpenCV calls,
 * dus zeer snel (< 1 ms voor 61 iteraties).
 */
function findBestSaFactor(_img, userTarget) {
  if (!lastResult || !lastResult.blobs) return null;
  const blobs = lastResult.blobs;
  const minSingleArea = lastResult.minSingleArea ?? MIN_SINGLE_AREA;
  let best = null;
  for (let i = 0; i <= 60; i++) {
    const f = 0.60 + i * 0.01;
    const { total } = blobsToDetections(blobs, f, minSingleArea);
    const diff = Math.abs(total - userTarget);
    if (best === null || diff < best.diff) {
      best = { factor: Math.round(f * 100) / 100, predictedTotal: total, diff };
    }
  }
  return best;
}

/**
 * Verstuurt feedback naar Supabase. Gooit een error bij mislukken.
 * Stuurt ook alle blob-statistieken mee zodat we patronen in de fouten kunnen leren.
 */
async function sendFeedback({ appCount, userCount, saFactor, bestSaFactor, sharePhoto, image, notes }) {
  let imagePath = null;

  // Eerst foto uploaden als de gebruiker dat wil
  if (sharePhoto && image) {
    const blob = await imageToBlob(image, 1280);  // schaal naar max 1280px om bandbreedte te sparen
    const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}.jpg`;
    const uploadUrl = `${SUPABASE_URL}/storage/v1/object/feedback-photos/${filename}`;
    const upResp = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
        'Content-Type': 'image/jpeg',
        'x-upsert': 'false',
      },
      body: blob,
    });
    if (!upResp.ok) {
      const txt = await upResp.text();
      throw new Error('Foto upload faalde: ' + txt);
    }
    imagePath = filename;
  }

  // Stats van de laatste telling (kan null zijn voor edge cases)
  const stats = lastResult?.stats || {};

  // Daarna de feedback-rij insteken (met alle blob-stats voor analytics)
  const insertUrl = `${SUPABASE_URL}/rest/v1/feedback`;
  const resp = await fetch(insertUrl, {
    method: 'POST',
    headers: {
      'apikey': SUPABASE_ANON_KEY,
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
      'Content-Type': 'application/json',
      'Prefer': 'return=minimal',
    },
    body: JSON.stringify({
      app_count: appCount,
      user_count: userCount,
      sa_factor: saFactor,
      best_sa_factor: bestSaFactor ?? null,
      image_path: imagePath,
      user_agent: navigator.userAgent.slice(0, 250),
      notes: notes ?? null,
      // Extra blob-features
      num_blobs: stats.num_blobs ?? null,
      num_singles: stats.num_singles ?? null,
      num_small_clumps: stats.num_small_clumps ?? null,
      num_medium_clumps: stats.num_medium_clumps ?? null,
      num_large_clumps: stats.num_large_clumps ?? null,
      single_area_estimate: stats.single_area_estimate ?? null,
      total_dark_area: stats.total_dark_area ?? null,
      largest_blob_area: stats.largest_blob_area ?? null,
      median_blob_area: stats.median_blob_area ?? null,
      image_width: lastResult?.imageWidth ?? null,
      image_height: lastResult?.imageHeight ?? null,
      app_version: APP_VERSION,
    }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error('Feedback insert faalde: ' + txt);
  }
}

/**
 * Schaal een Image naar maximaal maxDim pixels (langste zijde) en geef terug als JPEG Blob.
 */
function imageToBlob(img, maxDim = 1280) {
  return new Promise((resolve, reject) => {
    const w0 = img.naturalWidth, h0 = img.naturalHeight;
    const scale = Math.min(1, maxDim / Math.max(w0, h0));
    const w = Math.round(w0 * scale), h = Math.round(h0 * scale);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    c.getContext('2d').drawImage(img, 0, 0, w, h);
    c.toBlob(blob => {
      if (blob) resolve(blob);
      else reject(new Error('Canvas toBlob faalde'));
    }, 'image/jpeg', 0.85);
  });
}
