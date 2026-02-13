# Windows Playback Performance — Benchmark Guide

This document explains how to **verifiably measure** the improvements made to
Windows video playback (seek latency, frame-drop rate, decoder reset cost).

All measurements run inside the live application — no external tooling needed
beyond Chrome/Electron DevTools.

---

## Quick Start

1. Run the app in dev mode:
   ```
   cd apps/app
   npm start
   ```
2. Open DevTools: `Ctrl+Shift+I` → **Console** tab
3. Load a video clip onto the timeline
4. Follow the steps below

---

## Test 1 — Seek Latency (most important)

**What it measures:** How long each seek takes from the moment the worker receives
the message to when `seekDone` is posted back.

### Run in Console (after loading a clip):

```js
// Step 1: Reset stats so you start from zero
window.__apexResetSeekStats();

// Step 2: Scrub the timeline quickly for ~5 seconds, then stop

// Step 3: Read total seeks sent to the worker
const stats = window.__apexSeekStats();
console.table(stats);
// Expected on "after" build during fast scrub:
//   total: ~5-15  (one per 60ms window, collapsed by debounce)
//   fast:  ~5-15
//   accurate: 1-3 (settle seeks only)

// Step 4: Read actual worker-side seek durations
const measures = performance.getEntriesByType("measure")
  .filter(m => m.name.startsWith("seek-latency"));
const durations = measures.map(m => m.duration);
const avg = durations.reduce((a,b)=>a+b,0) / durations.length;
const max = Math.max(...durations);
console.log(`Seek count: ${durations.length}`);
console.log(`Avg seek latency: ${avg.toFixed(1)}ms`);
console.log(`Max seek latency: ${max.toFixed(1)}ms`);
```

### Expected results:

| Metric | BEFORE fixes | AFTER fixes |
|---|---|---|
| Seeks during 5s scrub | 50–300 | 5–15 |
| Avg seek latency | 80–400ms | 20–80ms |
| Max seek latency | 200–800ms | 60–150ms |

---

## Test 2 — GPU Decoder Reset Cost (Fix 2)

**What it measures:** How long `resetAndConfigureDecoders()` takes. Previously
this ran on every loop iteration; now it runs once per seek.

```js
// After scrubbing for a few seconds:
const resets = performance.getEntriesByType("measure")
  .filter(m => m.name.startsWith("gpu-reset-"));

const durations = resets.map(m => m.duration);
const avg = durations.reduce((a,b)=>a+b,0) / durations.length;
console.log(`GPU resets triggered: ${resets.length}`);
console.log(`Avg reset time: ${avg.toFixed(1)}ms`);
console.log(`Total reset overhead: ${durations.reduce((a,b)=>a+b,0).toFixed(1)}ms`);
```

### Expected results:

| Metric | BEFORE | AFTER |
|---|---|---|
| Resets per seek | 1–4 (per loop iteration) | 1 (upfront only) |
| Avg reset time | 10–50ms (Windows D3D11VA) | 10–50ms (same, but called less) |
| Total overhead for 5s scrub | 500ms–4000ms | 50–150ms |

---

## Test 3 — Seek Debounce Verification (Fix 3)

**What it measures:** The ratio of `focusFrame` updates to actual worker seeks.
Without debouncing, these are 1:1. With debouncing, ~60ms collapses them.

```js
// Reset stats
window.__apexResetSeekStats();

// Hold and drag the timeline scrubber for exactly 3 seconds
// (count "one-thousand-one, one-thousand-two, one-thousand-three")

// Then read:
const s = window.__apexSeekStats();
console.log("Worker seeks in 3s:", s.total);
// BEFORE: 60–180 seeks (one per focusFrame change at 60fps)
// AFTER:  ~3–8 seeks (one per 60ms debounce window)
```

---

## Test 4 — Playback Frame Drop Rate (Fix 4 + Fix 6)

**What it measures:** Whether frames are being dropped during playback by
watching the `lastRenderedFrame` counter advance without gaps.

```js
// Patch the render pipeline to count frame drops
let lastFrame = -1;
let drops = 0;
let total = 0;

// Intercept console or use the Performance timeline:
// Press Play, then after 10 seconds press Pause and run:
const entries = performance.getEntriesByType("measure")
  .filter(m => m.name.startsWith("seek-latency"));

// Better: use the DevTools Performance tab
// 1. Click Record in Performance tab
// 2. Press Play in the app for 10 seconds
// 3. Stop recording
// 4. Look at "Frames" row — green bars = smooth, red gaps = drops
// BEFORE: visible red gaps every 1-3 seconds on Windows
// AFTER:  continuous green bars
```

---

## Test 5 — DevTools Performance Timeline (Visual Proof)

This is the most visual test:

1. Open DevTools → **Performance** tab
2. Click the ⏺ Record button
3. Press **Play** in the Apex Studio timeline
4. Let it play for **10 seconds**
5. Click **Stop**
6. In the flame chart, look at:
   - **Frames** row: should be a continuous green bar (AFTER), not intermittent (BEFORE)
   - **Main** thread: `requestAnimationFrame` callbacks should be evenly spaced
   - **Worker** thread: look for `handleSeek` and `resetAndConfigureDecoders` — they should appear rarely (only on play start), not repeatedly

---

## Test 6 — Windows GPU Path Verification (Fix 5)

Verify D3D11 is active (Windows only):

```js
// In DevTools console:
const gpuInfo = await new Promise(r => {
  // Electron exposes this via ipcRenderer if you have access,
  // otherwise check chrome://gpu in a browser window
});

// Or open a new Electron window and navigate to:
// chrome://gpu
// Look for:
//   Graphics Feature Status → Video Decode: Hardware accelerated
//   GL_RENDERER: should mention D3D11 or ANGLE (Direct3D 11)
//   NOT: "Software only, hardware acceleration unavailable"
```

Alternatively, check the app logs on startup — the GPU info is printed to
console when `HardwareAccelerationModule` runs if you enable logging.

---

## Automated Before/After Comparison

If you want to compare the exact same scrub session before and after:

### BEFORE (revert just the debounce to compare seek counts):
```js
// Temporarily disable debounce to simulate old behavior:
// (don't do this in production, just for testing)
window.__apexResetSeekStats();
// Manually call seek 50 times rapidly:
for (let i = 0; i < 50; i++) {
  setTimeout(() => {
    // This simulates what used to happen every frame
    console.log("Would have called worker seek:", i);
  }, i * 16);
}
```

### Summary Table to Fill In:

| Test | Before | After | Improvement |
|---|---|---|---|
| Seeks per 5s scrub | _____ | _____ | _____ |
| Avg seek latency | _____ | _____ | _____ |
| GPU resets per seek | _____ | _____ | _____ |
| Frame drops in 10s | _____ | _____ | _____ |
| Scrub responsiveness | subjective | subjective | _____ |
