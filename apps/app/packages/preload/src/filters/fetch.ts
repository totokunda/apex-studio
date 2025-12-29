import fs from "node:fs";
import { join, basename, extname, relative } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
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
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const publicPath = join(
    __dirname,
    "..",
    "..",
    "..",
    "packages",
    "renderer",
    "public",
    "filters",
  );
  const smallBasePath = join(publicPath, "small");
  const fullBasePath = join(publicPath, "full");
  const exampleBasePath = join(publicPath, "examples");
  const exampleFiles = await findPngFiles(exampleBasePath, exampleBasePath);
  // Find all PNG files in the small directory
  const smallFiles = await findPngFiles(smallBasePath, smallBasePath);

  for (const relPath of smallFiles) {
    const fileName = basename(relPath, extname(relPath));
    const id = fileName;
    const name = snakeCaseToTitleCase(fileName);
    const smallPath = join(smallBasePath, relPath);
    const fullPath = join(fullBasePath, relPath);
    let examplePath = join(exampleBasePath, relPath);
    // remove everything before public/
    const exampleAssetUrl = pathToFileURL(examplePath).href;
    examplePath = examplePath.replace(publicPath, "filters");
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
