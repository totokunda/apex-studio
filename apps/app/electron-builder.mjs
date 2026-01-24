import pkg from "./package.json" with { type: "json" };
import mapWorkspaces from "@npmcli/map-workspaces";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { existsSync } from "node:fs";

const publishTimeoutMs = (() => {
  // GitHub uploads can stall on slow/unstable networks; electron-publish uses this as a request/socket timeout.
  // Override per environment if needed.
  const raw = process.env.ELECTRON_PUBLISH_TIMEOUT_MS;
  const fallback = 15 * 60 * 1000; // 15 minutes
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : fallback;
})();

// Determine if Python API bundle exists
const pythonApiBundlePath = join(process.cwd(), "python-api-bundle");
const hasPythonBundle = existsSync(pythonApiBundlePath);

export default /** @type import('electron-builder').Configuration */
({
  appId: "com.apex.studio",
  productName: "Apex Studio",
  
  directories: {
    output: "dist",
    buildResources: "buildResources",
  },
  
  generateUpdatesFilesForAllChannels: true,
  
  // Include Python API bundle in extraResources
  extraResources: hasPythonBundle ? [
    {
      from: "python-api-bundle",
      to: "python-api",
      filter: ["**/*"],
    },
  ] : [],
  
  /**
   * It is recommended to avoid using non-standard characters such as spaces in artifact names,
   * as they can unpredictably change during deployment, making them impossible to locate and download for update.
   */
  // Use `${name}` (package.json name) to avoid spaces in filenames.
  // NOTE: electron-builder doesn't define a `${target}` macro at this level.
  // Windows target-specific names are configured under `win.target` below.
  artifactName: "${name}-${version}-${os}-${arch}.${ext}",
  
  files: [
    "LICENSE*",
    pkg.main,
    "!node_modules/@app/**",
    ...(await getListOfFilesFromEachWorkspace()),
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
      "python-api/**/*.pyc",
      "python-api/**/__pycache__/**",
    ],
    notarize: process.env.APPLE_ID && process.env.APPLE_APP_PASSWORD ? {
      teamId: process.env.APPLE_TEAM_ID,
    } : false,
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
    signDlls: true,
    // Code signing configuration (requires WINDOWS_CERT_FILE environment variable)
    sign: process.env.WINDOWS_CERT_FILE ? "./scripts/sign-windows.js" : undefined,
    signingHashAlgorithms: ["sha256"],
    publisherName: process.env.WINDOWS_PUBLISHER_NAME || "Apex Studio",
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

  // Windows "portable" target options
  portable: {
    artifactName: "${name}-${version}-${os}-${arch}-portable.${ext}",
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
    ...(process.env.AWS_S3_BUCKET ? [{
      provider: "s3",
      bucket: process.env.AWS_S3_BUCKET,
      region: process.env.AWS_REGION || "us-east-1",
      timeout: publishTimeoutMs,
    }] : []),
  ],
  
  // Auto-update configuration
  autoUpdates: {
    allowDowngrade: false,
    allowPrerelease: false,
  },
});

/**
 * By default, electron-builder copies each package into the output compilation entirety,
 * including the source code, tests, configuration, assets, and any other files.
 *
 * So you may get compiled app structure like this:
 * ```
 * app/
 * ├── node_modules/
 * │   └── workspace-packages/
 * │       ├── package-a/
 * │       │   ├── src/            # Garbage. May be safely removed
 * │       │   ├── dist/
 * │       │   │   └── index.js    # Runtime code
 * │       │   ├── vite.config.js  # Garbage
 * │       │   ├── .env            # some sensitive config
 * │       │   └── package.json
 * │       ├── package-b/
 * │       ├── package-c/
 * │       └── package-d/
 * ├── packages/
 * │   └── entry-point.js
 * └── package.json
 * ```
 *
 * To prevent this, we read the "files"
 * property from each package's package.json
 * and add all files that do not match the patterns to the exclusion list.
 *
 * This way,
 * each package independently determines which files will be included in the final compilation and which will not.
 *
 * So if `package-a` in its `package.json` describes
 * ```json
 * {
 *   "name": "package-a",
 *   "files": [
 *     "dist/**\/"
 *   ]
 * }
 * ```
 *
 * Then in the compilation only those files and `package.json` will be included:
 * ```
 * app/
 * ├── node_modules/
 * │   └── workspace-packages/
 * │       ├── package-a/
 * │       │   ├── dist/
 * │       │   │   └── index.js    # Runtime code
 * │       │   └── package.json
 * │       ├── package-b/
 * │       ├── package-c/
 * │       └── package-d/
 * ├── packages/
 * │   └── entry-point.js
 * └── package.json
 * ```
 */
async function getListOfFilesFromEachWorkspace() {
  /**
   * @type {Map<string, string>}
   */
  const workspaces = await mapWorkspaces({
    cwd: process.cwd(),
    pkg,
  });

  const allFilesToInclude = [];

  for (const [name, path] of workspaces) {
    const pkgPath = join(path, "package.json");
    const { default: workspacePkg } = await import(pathToFileURL(pkgPath), {
      with: { type: "json" },
    });

    let patterns = workspacePkg.files || ["dist/**", "package.json"];

    patterns = patterns.map((p) => join("node_modules", name, p));
    allFilesToInclude.push(...patterns);
  }

  return allFilesToInclude;
}
