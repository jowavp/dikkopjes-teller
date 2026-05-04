// Dikkopjes-teller service worker
// Cacht de app shell + OpenCV.js zodat alles offline werkt na het eerste bezoek.

const CACHE_VERSION = 'dikkopjes-v1';

// Wat lokaal aanwezig is en bij installatie meteen gecached wordt
const APP_SHELL = [
  './',
  './index.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  './apple-touch-icon.png',
  './favicon-32.png',
  './favicon-16.png',
];

// OpenCV.js komt van een CDN — we cachen het na de eerste fetch (runtime cache)
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then(cache => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  // Verwijder oude cache versies
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_VERSION).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const req = event.request;
  // Alleen GET requests cachen
  if (req.method !== 'GET') return;

  event.respondWith(
    caches.match(req).then(cached => {
      if (cached) return cached;
      return fetch(req).then(resp => {
        // Cache succesvolle responses (incl. OpenCV.js van CDN)
        if (resp && resp.status === 200 && (resp.type === 'basic' || resp.type === 'cors')) {
          const respClone = resp.clone();
          caches.open(CACHE_VERSION).then(cache => cache.put(req, respClone));
        }
        return resp;
      }).catch(() => {
        // Offline en niet in cache — fallback naar de app shell
        if (req.mode === 'navigate') return caches.match('./index.html');
      });
    })
  );
});
