const { existsSync } = require("node:fs");
const { join } = require("node:path");

// electron-builder loads config via CommonJS `require()`. Using `.cjs` avoids ESM issues.
const pkg = require("./package.json");
const path = require("node:path");


/** @type {import("electron-builder").Configuration} */
module.exports = {
  appId: "com.apex.studio",
  productName: "Apex Studio",

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
  artifactName: "${productName}-${version}-${os}-${arch}.${ext}",

  // Keep packaging minimal: include our entry point and only built artifacts from workspace packages.
  files: [
    "LICENSE*",
    pkg.main,

    // Exclude everything first, then explicitly include only built outputs.
    "!node_modules/@app/**",
    "node_modules/@app/**/dist/**",
    "node_modules/@app/**/package.json",

    // Size trims:
    // - renderer filter packs are huge; ship "small" only by default
    "!node_modules/@app/renderer/dist/filters/full/**",
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
    notarize:
      process.env.APPLE_ID && process.env.APPLE_APP_PASSWORD
        ? {
            teamId: process.env.APPLE_TEAM_ID,
          }
        : false,
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
    sign: false, // DMG signing is often not needed and can cause issues
  },

  // Windows configuration
  win: {
    target: [
      {
        target: "nsis",
        arch: ["x64"],
      },
      {
        target: "portable",
        arch: ["x64"],
      },
    ],
    icon: "buildResources/icon.ico",
  },

  nsis: {
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
      owner: process.env.GITHUB_OWNER || "apex-studio",
      repo: process.env.GITHUB_REPO || "apex-studio",
      releaseType: "release",
      publishAutoUpdate: true,
    },
    // Optional: S3 for faster downloads
    ...(process.env.AWS_S3_BUCKET
      ? [
          {
            provider: "s3",
            bucket: process.env.AWS_S3_BUCKET,
            region: process.env.AWS_REGION || "us-east-1",
          },
        ]
      : []),
  ],

  // Auto-update configuration
};


