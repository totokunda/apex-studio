import fs from "node:fs";
import { join, basename, extname, relative, dirname, resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createRequire } from "node:module";
// id should be the basename of the file

interface Filter {
  id: string;
  name: string;
  smallPath: string;
  fullPath: string;
  category: string;
  examplePath: string;
  exampleAssetUrl: string;
}

const require = createRequire(import.meta.url);

const tryResolveRendererStaticRoot = (): string | null => {
  const candidates: string[] = [];

  // Packaged builds (and some dev layouts) where node resolution works.
  try {
    const indexHtml = require.resolve("@app/renderer/dist/index.html");
    candidates.push(dirname(indexHtml));
  } catch {
    // ignore
  }

  // Monorepo dev layout: resolve relative to this module location.
  // This must work from both TS source (preload/src/...) and built output (preload/dist/...).
  try {
    const here = dirname(fileURLToPath(import.meta.url));
    candidates.push(resolve(here, "../../renderer/dist"));
    candidates.push(resolve(here, "../../renderer/public"));
    candidates.push(resolve(here, "../../../renderer/dist"));
    candidates.push(resolve(here, "../../../renderer/public"));
  } catch {
    // ignore
  }

  // Fallbacks relative to process CWD (varies across tooling).
  try {
    candidates.push(resolve(process.cwd(), "packages/renderer/dist"));
    candidates.push(resolve(process.cwd(), "packages/renderer/public"));
    candidates.push(resolve(process.cwd(), "apps/app/packages/renderer/dist"));
    candidates.push(resolve(process.cwd(), "apps/app/packages/renderer/public"));
  } catch {
    // ignore
  }

  // Pick the first candidate that actually contains the filters.
  for (const root of candidates) {
    try {
      if (fs.existsSync(join(root, "filters", "small"))) {
        return root;
      }
    } catch {
      // ignore
    }
  }

  console.warn(
    "[fetchFilters] Could not resolve renderer static root (expected to contain filters/*). Candidates:",
    candidates,
  );
  return null;
};

// Convert snake_case to Title Case
const snakeCaseToTitleCase = (str: string): string => {
  return str
    .split(/[_\-\+\s]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
};

// Recursively find all PNG files in a directory
const findPngFiles = async (
  dir: string,
  baseDir: string,
): Promise<string[]> => {
  const files: string[] = [];
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = join(dir, entry.name);
    if (entry.isDirectory()) {
      const nestedFiles = await findPngFiles(fullPath, baseDir);
      files.push(...nestedFiles);
    } else if (entry.isFile() && extname(entry.name).toLowerCase() === ".png") {
      files.push(relative(baseDir, fullPath));
    }
  }

  return files;
};




export const fetchFilters = async () => {
  const filters: Filter[] = [];
  // In packaged builds, the renderer assets live under '@app/renderer/dist'.
  // In dev, they're served by Vite and the source of truth is 'packages/renderer/public'.
  const rendererStaticRoot = tryResolveRendererStaticRoot();
  if (!rendererStaticRoot) {
    return [];
  }

  const filtersRoot = join(rendererStaticRoot, "filters");

  const smallBasePath = join(filtersRoot, "small");
  const fullBasePath = join(filtersRoot, "full");
  const exampleBasePath = join(filtersRoot, "examples");

  // If full/examples are excluded from packaging for size, fall back to small.
  const hasFull = fs.existsSync(fullBasePath);
  const hasExamples = fs.existsSync(exampleBasePath);

  // Find all PNG files in the small directory
  const smallFiles = await findPngFiles(smallBasePath, smallBasePath);

  for (const relPath of smallFiles) {
    const fileName = basename(relPath, extname(relPath));
    const id = fileName;
    const name = snakeCaseToTitleCase(fileName);
    const smallPath = join(smallBasePath, relPath);
    const fullPathCandidate = join(fullBasePath, relPath);
    const fullPath = hasFull && fs.existsSync(fullPathCandidate) ? fullPathCandidate : smallPath;

    const exampleFsPathCandidate = join(exampleBasePath, relPath);
    const exampleFsPath =
      hasExamples && fs.existsSync(exampleFsPathCandidate) ? exampleFsPathCandidate : smallPath;

    // Renderer-facing URL-ish path (used in <img src="...">).
    // Always use forward slashes.
    const examplePath = (
      hasExamples && fs.existsSync(exampleFsPathCandidate)
        ? `filters/examples/${relPath}`
        : `filters/small/${relPath}`
    ).replace(/\\/g, "/");

    const exampleAssetUrl = pathToFileURL(exampleFsPath).href;

    // Extract category from the immediate parent directory
    const dirPath = dirname(relPath);
    const categoryDir = basename(dirPath);
    const category = snakeCaseToTitleCase(categoryDir);

    filters.push({
      id,
      name,
      smallPath,
      fullPath,
      category,
      examplePath,
      exampleAssetUrl,
    });
  }

  return filters;
};
