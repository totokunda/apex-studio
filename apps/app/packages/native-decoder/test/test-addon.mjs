/**
 * Native decoder addon test suite.
 *
 * Run via:  npx electron --no-sandbox test/test-addon.mjs
 *
 * These tests verify the C++ N-API addon loads correctly, exports the
 * expected API surface, and (when a test fixture video exists) can
 * actually decode frames through SharedArrayBuffer.
 */

import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { strict as assert } from "node:assert";

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));
const addonDir = join(__dirname, "..", "build", "Release");
const addonPath = join(addonDir, "native_decoder.node");

let passed = 0;
let failed = 0;
let skipped = 0;

async function runTest(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    if (e === "SKIP") {
      skipped++;
      console.log(`  - ${name} (skipped)`);
    } else {
      failed++;
      console.error(`  \u2717 ${name}`);
      console.error(`    ${e?.message || e}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Test 1: Addon binary exists and loads
// ---------------------------------------------------------------------------

let addon;

await runTest("Addon binary loads without error", () => {
  if (!existsSync(addonPath)) {
    throw new Error(
      `Addon not found at ${addonPath}. Run "npm run build" in packages/native-decoder first.`
    );
  }
  addon = require(addonPath);
  assert.ok(addon, "require() returned a falsy value");
});

if (!addon) {
  console.error("\nCannot continue — addon failed to load.\n");
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Test 2: All 9 exported functions exist with correct types
// ---------------------------------------------------------------------------

const expectedExports = [
  "createDecoder",
  "configure",
  "seek",
  "iterate",
  "ackFrame",
  "cancelCurrent",
  "dispose",
  "disposeAll",
  "getCapabilities",
];

await runTest("All 9 exported functions exist", () => {
  for (const name of expectedExports) {
    assert.equal(
      typeof addon[name],
      "function",
      `Expected addon.${name} to be a function, got ${typeof addon[name]}`
    );
  }
});

// ---------------------------------------------------------------------------
// Test 3: getCapabilities returns valid structure
// ---------------------------------------------------------------------------

await runTest("getCapabilities() returns valid structure", () => {
  const caps = addon.getCapabilities();
  assert.ok(caps, "getCapabilities() returned falsy");
  assert.ok(
    Array.isArray(caps.hwAccelMethods),
    `hwAccelMethods should be an array, got ${typeof caps.hwAccelMethods}`
  );
  assert.equal(
    typeof caps.preferredMethod,
    "string",
    `preferredMethod should be a string, got ${typeof caps.preferredMethod}`
  );
  console.log(
    `    (hwAccel: [${caps.hwAccelMethods.join(", ")}], preferred: "${caps.preferredMethod}")`
  );
});

// ---------------------------------------------------------------------------
// Test 4: createDecoder returns unique positive numeric handles
// ---------------------------------------------------------------------------

await runTest("createDecoder() returns unique positive handles", () => {
  const h1 = addon.createDecoder();
  const h2 = addon.createDecoder();
  assert.equal(typeof h1, "number", "Handle 1 should be a number");
  assert.equal(typeof h2, "number", "Handle 2 should be a number");
  assert.ok(h1 > 0, `Handle 1 should be positive, got ${h1}`);
  assert.ok(h2 > 0, `Handle 2 should be positive, got ${h2}`);
  assert.notEqual(h1, h2, "Handles should be unique");

  // Clean up
  addon.dispose(h1);
  addon.dispose(h2);
});

// ---------------------------------------------------------------------------
// Test 5: dispose and disposeAll don't throw
// ---------------------------------------------------------------------------

await runTest("dispose() and disposeAll() don't throw", () => {
  const h = addon.createDecoder();
  addon.dispose(h);
  // Double-dispose should be safe (no-op or ignore)
  addon.dispose(h);

  // disposeAll with no active decoders
  addon.disposeAll();
});

// ---------------------------------------------------------------------------
// Test 6: ackFrame and cancelCurrent on invalid handles are safe
// ---------------------------------------------------------------------------

await runTest("ackFrame/cancelCurrent on invalid handles don't crash", () => {
  // These should silently fail or be no-ops, not segfault
  addon.ackFrame(999999, 0);
  addon.cancelCurrent(999999);
});

// ---------------------------------------------------------------------------
// Test 7 (conditional): Decode a single frame from test fixture
// ---------------------------------------------------------------------------

const testVideo = join(__dirname, "fixtures", "test.mp4");
const hasTestVideo = existsSync(testVideo);

await runTest(
  "Seek to frame 0 and receive onFrame + onReady callbacks",
  async () => {
    if (!hasTestVideo) throw "SKIP";

    const width = 320;
    const height = 240;
    const bytesPerFrame = width * height * 4;
    const poolSize = 4;
    const bufferPool = Array.from(
      { length: poolSize },
      () => new SharedArrayBuffer(bytesPerFrame)
    );

    const handle = addon.createDecoder();
    let readyFired = false;
    let frameData = null;

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("Timeout waiting for onReady/onFrame")),
        10000
      );

      addon.configure(handle, {
        filePath: testVideo,
        width,
        height,
        bufferPool,
        onFrame: (bufferIndex, w, h, timestamp, duration, requestId) => {
          if (!frameData) {
            frameData = { bufferIndex, w, h, timestamp, duration, requestId };
            addon.ackFrame(handle, bufferIndex);
            clearTimeout(timeout);
            resolve();
          }
        },
        onError: (msg) => {
          clearTimeout(timeout);
          reject(new Error(`Decoder error: ${msg}`));
        },
        onReady: () => {
          readyFired = true;
          // Seek to the very first frame
          addon
            .seek(handle, 0, false, 1)
            .catch((e) => reject(new Error(`Seek failed: ${e}`)));
        },
      });
    });

    assert.ok(readyFired, "onReady should have fired");
    assert.ok(frameData, "onFrame should have fired at least once");
    assert.equal(frameData.w, width, "Frame width should match");
    assert.equal(frameData.h, height, "Frame height should match");
    assert.ok(frameData.bufferIndex >= 0, "bufferIndex should be non-negative");
    assert.ok(
      frameData.bufferIndex < poolSize,
      "bufferIndex should be within pool range"
    );

    // Verify the buffer has non-zero data (not an empty black frame for most videos)
    const view = new Uint8Array(bufferPool[frameData.bufferIndex]);
    let nonZero = 0;
    for (let i = 0; i < Math.min(1000, view.length); i++) {
      if (view[i] !== 0) nonZero++;
    }
    assert.ok(nonZero > 0, "Frame buffer should contain non-zero RGBA data");

    addon.dispose(handle);
  }
);

// ---------------------------------------------------------------------------
// Test 8 (conditional): Iterate a 1-second range, verify multiple frames
// ---------------------------------------------------------------------------

await runTest(
  "Iterate 1-second range yields multiple frames with increasing timestamps",
  async () => {
    if (!hasTestVideo) throw "SKIP";

    const width = 320;
    const height = 240;
    const bytesPerFrame = width * height * 4;
    const poolSize = 4;
    const bufferPool = Array.from(
      { length: poolSize },
      () => new SharedArrayBuffer(bytesPerFrame)
    );

    const handle = addon.createDecoder();
    const frames = [];

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("Timeout waiting for iteration")),
        15000
      );

      addon.configure(handle, {
        filePath: testVideo,
        width,
        height,
        bufferPool,
        onFrame: (bufferIndex, w, h, timestamp, duration, requestId) => {
          frames.push({ bufferIndex, timestamp, duration });
          // Ack so the next frame can be delivered
          addon.ackFrame(handle, bufferIndex);
        },
        onError: (msg) => {
          clearTimeout(timeout);
          reject(new Error(`Decoder error: ${msg}`));
        },
        onReady: () => {
          // Iterate from 0s to 1s
          addon
            .iterate(handle, 0, 1_000_000, 2)
            .then(() => {
              clearTimeout(timeout);
              resolve();
            })
            .catch((e) => {
              clearTimeout(timeout);
              reject(new Error(`Iterate failed: ${e}`));
            });
        },
      });
    });

    assert.ok(
      frames.length >= 2,
      `Expected at least 2 frames from 1s iteration, got ${frames.length}`
    );

    // Verify timestamps are non-decreasing
    for (let i = 1; i < frames.length; i++) {
      assert.ok(
        frames[i].timestamp >= frames[i - 1].timestamp,
        `Frame ${i} timestamp (${frames[i].timestamp}) should be >= frame ${i - 1} (${frames[i - 1].timestamp})`
      );
    }

    // Verify all bufferIndex values are in valid range
    for (const f of frames) {
      assert.ok(f.bufferIndex >= 0 && f.bufferIndex < poolSize,
        `bufferIndex ${f.bufferIndex} out of range [0, ${poolSize})`);
    }

    console.log(`    (received ${frames.length} frames)`);

    addon.dispose(handle);
  }
);

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\n${passed + failed + skipped} tests: ${passed} passed, ${failed} failed, ${skipped} skipped\n`);

if (!hasTestVideo) {
  console.log(
    "Note: Place a test video at test/fixtures/test.mp4 to enable decode tests.\n"
  );
}

process.exit(failed > 0 ? 1 : 0);
