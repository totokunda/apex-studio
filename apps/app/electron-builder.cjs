const { existsSync } = require("node:fs");
const { join } = require("node:path");

// electron-builder loads config via CommonJS `require()`. Using `.cjs` avoids ESM issues.
const pkg = require("./package.json");
const path = require("node:path");

const publishTimeoutMs = (() => {
  // GitHub uploads can stall on slow/unstable networks; electron-publish uses this as a request/socket timeout.
  // Override per environment if needed.
  const raw = process.env.ELECTRON_PUBLISH_TIMEOUT_MS;
  const fallback = 15 * 60 * 1000; // 15 minutes
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : fallback;
})();

const bundleFullFilters = process.env.APEX_BUNDLE_FILTERS_FULL === "1";

// electron-builder v26: `mac.notarize` enables notarization when truthy.
// We disable notarization when we don't have the required environment variables.
const hasMacNotarizeEnv =
  // App Store Connect API key flow
  (process.env.APPLE_API_KEY && process.env.APPLE_API_KEY_ID && process.env.APPLE_API_ISSUER) ||
  // Apple ID + app-specific password flow
  (process.env.APPLE_ID && process.env.APPLE_APP_SPECIFIC_PASSWORD && process.env.APPLE_TEAM_ID) ||
  // Keychain profile flow
  (process.env.APPLE_KEYCHAIN && process.env.APPLE_KEYCHAIN_PROFILE);

/** @type {import("electron-builder").Configuration} */
module.exports = {
  appId: "com.apex.studio",
  productName: "Apex Studio",

  // Keep ASAR enabled (default). We intentionally avoid broad `asarUnpack` globs,
  // because @electron/asar uses minimatch internally and can throw "pattern is too long"
  // when unpack patterns expand into huge brace patterns on large trees.

  directories: {
    output: "dist",
    buildResources: "buildResources",
  },

  generateUpdatesFilesForAllChannels: true,

  // Extra binaries shipped alongside the app (available at runtime under `process.resourcesPath`)
  // - python-api: optional (bundled by separate build step)
  extraResources: (() => {
    const out = [];
    const pythonApiDir = path.join(__dirname, "buildResources", "python-api");
    if (existsSync(pythonApiDir)) out.push({ from: pythonApiDir, to: "python-api" });
    return out;
  })(),

  /**
   * It is recommended to avoid using non-standard characters such as spaces in artifact names,
   * as they can unpredictably change during deployment, making them impossible to locate and download for update.
   */
  // Use `${name}` (package.json name) to avoid spaces in filenames.
  // NOTE: electron-builder doesn't define a `${target}` macro at this level.
  // Windows target-specific names are configured under `win.target` below.
  artifactName: "${name}-${version}-${os}-${arch}.${ext}",

  // Keep packaging minimal: include our entry point and only built artifacts from workspace packages.
  files: [
    "LICENSE*",
    pkg.main,

    // Include built outputs from workspace packages.
    "node_modules/@app/*/dist/**",
    "node_modules/@app/*/package.json",

    // Size trims:
    // - renderer filter packs are huge; ship "small" and "examples" by default
    ...(bundleFullFilters ? [] : ["!node_modules/@app/renderer/dist/filters/full/**"]),
    // - never ship renderer source/public trees (dev-only); dist/** is the runtime payload
    "!node_modules/@app/renderer/public/**",
    "!node_modules/@app/renderer/src/**",
    "!node_modules/@app/renderer/scripts/**",
    "!node_modules/@app/renderer/vite.config.*",
    "!node_modules/@app/renderer/tailwind.config.*",
    "!node_modules/@app/renderer/tsconfig*.json",
    "!node_modules/@app/renderer/components.json",
    "!node_modules/@app/renderer/index.html",
    // - ffprobe-static is required on all targets; include wrapper + binaries
    "node_modules/ffprobe-static/**",
    "!node_modules/ffprobe-static/tests/**",
    // - model demo media is optional and large; download later if needed
  ],

  // macOS configuration
  mac: {
    category: "public.app-category.video",
    target: [
      {
        target: "dmg",
        arch: ["x64", "arm64"],
      },
      {
        target: "zip",
        arch: ["x64", "arm64"],
      },
    ],
    icon: "buildResources/icon.icns",
    hardenedRuntime: true,
    gatekeeperAssess: false,
    entitlements: "buildResources/entitlements.mac.plist",
    entitlementsInherit: "buildResources/entitlements.mac.plist",
    signIgnore: [
      // Skip signing certain Python files that may cause issues
      "^python-api/.*\\.pyc$",
      "^python-api/.*/__pycache__/.*",
    ],
    // Enable notarization only when required env vars are present.
    notarize: Boolean(hasMacNotarizeEnv),
  },
  dmg: {
    contents: [
      {
        x: 130,
        y: 220,
      },
      {
        x: 410,
        y: 220,
        type: "link",
        path: "/Applications",
      },
    ],
    sign: true, // DMG signing is often not needed and can cause issues
  },

  // Windows configuration
  win: {
    target: [
      {
        target: "nsis",
        arch: ["x64"],
      },
    ],
    icon: "buildResources/icon.ico",
  },

  nsis: {
    artifactName: "${name}-${version}-${os}-${arch}-setup.${ext}",
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    perMachine: false,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    shortcutName: "Apex Studio",
    installerIcon: "buildResources/icon.ico",
    uninstallerIcon: "buildResources/icon.ico",
    installerHeaderIcon: "buildResources/icon.ico",
    // Include license
    license: "LICENSE",
    // Custom NSIS script for Python runtime setup
    include: "buildResources/installer.nsh",
  },

  // Linux configuration
  linux: {
    target: [
      {
        target: "deb",
        arch: ["x64"],
      },
      {
        target: "AppImage",
        arch: ["x64"],
      },
      {
        target: "tar.gz",
        arch: ["x64"],
      },
    ],
    category: "Video",
    icon: "buildResources/icon.png",
    description: "Apex Studio - AI Video Generation",
    maintainer: "Apex Studio Team <support@apex.studio>",
  },

  deb: {
    depends: [
      "libgtk-3-0",
      "libnotify4",
      "libnss3",
      "libxss1",
      "libxtst6",
      "xdg-utils",
      "libatspi2.0-0",
      "libuuid1",
      "libsecret-1-0",
    ],
    afterInstall: "buildResources/linux/postinst.sh",
    afterRemove: "buildResources/linux/postrm.sh",
  },

  appImage: {
    artifactName: "${productName}-${version}-${arch}.${ext}",
  },

  // Publishing configuration
  publish: [
    {
      provider: "github",
      owner: process.env.GITHUB_OWNER || "totokunda",
      repo: process.env.GITHUB_REPO || "apex-studio",
      releaseType: "release",
      publishAutoUpdate: true,
      timeout: publishTimeoutMs,
    },
    // Optional: S3 for faster downloads
    ...(process.env.AWS_S3_BUCKET
      ? [
          {
            provider: "s3",
            bucket: process.env.AWS_S3_BUCKET,
            region: process.env.AWS_REGION || "us-east-1",
            timeout: publishTimeoutMs,
          },
        ]
      : []),
  ],

  // Auto-update configuration
};


