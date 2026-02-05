from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src" / "quantize" / "_ggml_quants.cpp"
    if not src.exists():
        raise FileNotFoundError(src)

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required to build the extension") from e

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = repo_root / "src" / "quantize" / f"_ggml_quants{ext_suffix}"

    py_include = sysconfig.get_paths()["include"]
    np_include = np.get_include()

    ggml_lib_dir = repo_root / "llama-b7902"
    if not ggml_lib_dir.exists():
        raise FileNotFoundError(ggml_lib_dir)

    cmd = [
        "g++",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        f"-I{py_include}",
        f"-I{np_include}",
        "-o",
        str(out),
        str(src),
        f"-L{ggml_lib_dir}",
        "-lggml-base",
        "-Wl,-rpath,$ORIGIN/../../llama-b7902",
    ]

    subprocess.check_call(cmd, cwd=repo_root)
    print(f"Built: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
