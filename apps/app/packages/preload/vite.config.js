import { getChromeMajorVersion } from "@app/electron-versions";
import ts from "typescript";
import fs from "node:fs";
import path from "node:path";

export default /**
 * @type {import('vite').UserConfig}
 * @see https://vitejs.dev/config/
 */
({
  build: {
    ssr: true,
    sourcemap: "inline",
    outDir: "dist",
    target: `chrome${getChromeMajorVersion()}`,
    assetsDir: ".",
    lib: {
      // Use named entries so output filenames are stable and match package.json exports:
      // - "./exposed.mjs" -> "./dist/exposed.mjs"
      // - "." (main/module) -> "./dist/_virtual_browser.mjs"
      entry: {
        exposed: "src/exposed.ts",
        _virtual_browser: "virtual:browser.js",
      },
    },
    rollupOptions: {
      output: [
        {
          // ESM preload scripts must have the .mjs extension
          // https://www.electronjs.org/docs/latest/tutorial/esm#esm-preload-scripts-must-have-the-mjs-extension
          entryFileNames: "[name].mjs",
        },
      ],
    },
    emptyOutDir: true,
    reportCompressedSize: false,
  },
  plugins: [mockExposed(), handleHotReload()],
});

/**
 * This plugin creates a browser (renderer) version of `preload` package.
 * Basically, it just read all nominals you exported from package and define it as globalThis properties
 * expecting that real values were exposed by `electron.contextBridge.exposeInMainWorld()`
 *
 * Example:
 * ```ts
 * // index.ts
 * export const someVar = 'my-value';
 * ```
 *
 * Output
 * ```js
 * // _virtual_browser.mjs
 * export const someVar = globalThis[<hash>] // 'my-value'
 * ```
 */
function mockExposed() {
  const virtualModuleId = "virtual:browser.js";
  const resolvedVirtualModuleId = "\0" + virtualModuleId;

  const resolveExportTarget = (fromFile, spec) => {
    // Map `./foo.js` to `./foo.ts` / `./foo.tsx` / `./foo.js` etc.
    // This is needed because our TS source uses `.js` specifiers for ESM,
    // but during this build step we’re enumerating exports from source files.
    const baseDir = path.dirname(fromFile);
    const raw = spec.startsWith("file:")
      ? new URL(spec).pathname
      : path.resolve(baseDir, spec);

    const candidates = [];
    if (path.extname(raw)) {
      // If the spec includes an extension, try swapping it.
      const stem = raw.replace(/\.[^.]+$/, "");
      candidates.push(`${stem}.ts`, `${stem}.tsx`, `${stem}.js`, `${stem}.mjs`);
      candidates.push(raw);
    } else {
      candidates.push(`${raw}.ts`, `${raw}.tsx`, `${raw}.js`, `${raw}.mjs`);
      candidates.push(path.join(raw, "index.ts"));
      candidates.push(path.join(raw, "index.tsx"));
      candidates.push(path.join(raw, "index.js"));
      candidates.push(path.join(raw, "index.mjs"));
    }

    for (const c of candidates) {
      try {
        if (fs.existsSync(c)) return c;
      } catch {}
    }
    return null;
  };

  const collectExportNames = (entryFile) => {
    const seen = new Set();
    const out = new Set();

    /** @param {string} file */
    const visit = (file) => {
      if (!file) return;
      const abs = path.resolve(file);
      if (seen.has(abs)) return;
      seen.add(abs);

      let sourceText = "";
      try {
        sourceText = fs.readFileSync(abs, "utf8");
      } catch {
        return;
      }

      const sf = ts.createSourceFile(
        abs,
        sourceText,
        ts.ScriptTarget.ESNext,
        true,
        abs.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
      );

      const addName = (n) => {
        if (!n) return;
        if (n === "default") return;
        out.add(n);
      };

      sf.forEachChild((node) => {
        // export const foo / export function foo / export class Foo
        if (
          (ts.isVariableStatement(node) ||
            ts.isFunctionDeclaration(node) ||
            ts.isClassDeclaration(node) ||
            ts.isInterfaceDeclaration(node) ||
            ts.isEnumDeclaration(node) ||
            ts.isTypeAliasDeclaration(node)) &&
          node.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword)
        ) {
          // Type-only exports don’t exist at runtime; skip them
          if (
            ts.isInterfaceDeclaration(node) ||
            ts.isTypeAliasDeclaration(node)
          ) {
            return;
          }
          if (ts.isVariableStatement(node)) {
            for (const decl of node.declarationList.declarations) {
              if (ts.isIdentifier(decl.name)) addName(decl.name.text);
            }
          } else {
            if (node.name && ts.isIdentifier(node.name)) addName(node.name.text);
          }
          return;
        }

        // export { a, b as c } [from "./x"]
        if (ts.isExportDeclaration(node)) {
          if (node.isTypeOnly) return;
          const spec = node.moduleSpecifier?.text;

          if (node.exportClause && ts.isNamedExports(node.exportClause)) {
            for (const el of node.exportClause.elements) {
              // `export { default as foo } from ...` is legal
              addName(el.name.text);
            }
            // If it’s a re-export, we don’t need to visit the module for names,
            // because the clause tells us the names.
            return;
          }

          // export * as ns from "./x"
          if (node.exportClause && ts.isNamespaceExport(node.exportClause)) {
            addName(node.exportClause.name.text);
            return;
          }

          // export * from "./x"
          if (spec) {
            const resolved = resolveExportTarget(abs, spec);
            if (resolved) visit(resolved);
          }
          return;
        }

        // export default ... (not used by this package, but handle just in case)
        if (ts.isExportAssignment(node)) {
          if (!node.isExportEquals) {
            // `export default`
            // We intentionally skip default because our shim only needs named exports.
          }
        }
      });
    };

    visit(entryFile);
    return Array.from(out).sort();
  };

  return {
    name: "electron-main-exposer",
    resolveId(id) {
      if (id.endsWith(virtualModuleId)) {
        return resolvedVirtualModuleId;
      }
    },
    async load(id) {
      if (id === resolvedVirtualModuleId) {
        const entryAbs = new URL("./src/index.ts", import.meta.url);
        const exportedNames = collectExportNames(entryAbs.pathname);
        return exportedNames.reduce((s, key) => {
          return (
            s +
            (key === "default"
              ? `export default globalThis['${btoa(key)}'];\n`
              : `export const ${key} = globalThis['${btoa(key)}'];\n`)
          );
        }, "");
      }
    },
  };
}

/**
 * Implement Electron webview reload when some file was changed
 * @return {import('vite').Plugin}
 */
function handleHotReload() {
  /** @type {import('vite').ViteDevServer|null} */
  let rendererWatchServer = null;

  return {
    name: "@app/preload-process-hot-reload",

    config(config, env) {
      if (env.mode !== "development") {
        return;
      }

      const rendererWatchServerProvider = config.plugins.find(
        (p) => p.name === "@app/renderer-watch-server-provider",
      );
      if (!rendererWatchServerProvider) {
        throw new Error("Renderer watch server provider not found");
      }

      rendererWatchServer =
        rendererWatchServerProvider.api.provideRendererWatchServer();

      return {
        build: {
          watch: {},
        },
      };
    },

    writeBundle() {
      if (!rendererWatchServer) {
        return;
      }

      rendererWatchServer.ws.send({
        type: "full-reload",
      });
    },
  };
}
