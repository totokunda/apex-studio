from __future__ import annotations

import platform
import subprocess
import sysconfig
from pathlib import Path
import argparse

def main(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1].parent
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
    py_lib_dir = sysconfig.get_config_var("LIBDIR")

    ggml_lib_dir = repo_root / f"llama-b{args.ggml_version}"
    if not ggml_lib_dir.exists():
        raise FileNotFoundError(ggml_lib_dir)

    is_darwin = platform.system() == "Darwin"

    # On macOS, use -bundle and -undefined dynamic_lookup for Python extensions
    # On Linux, use -shared and link flags
    if is_darwin:
        # @loader_path = dir containing the .so; quantize/ -> ../../llama-b8027
        rpath = f"@loader_path/../../llama-b{args.ggml_version}"
        link_args = [
            "-bundle",
            "-undefined",
            "dynamic_lookup",
            f"-Wl,-rpath,{rpath}",
        ]
    else:
        rpath = f"$ORIGIN/../../llama-b{args.ggml_version}"
        link_args = [
            "-shared",
            f"-Wl,-rpath,{rpath}",
        ]

    cmd = [
        "clang++" if is_darwin else "g++",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fPIC",
        "-pthread",
        f"-I{py_include}",
        f"-I{np_include}",
        "-o",
        str(out),
        str(src),
        f"-L{py_lib_dir}",
        f"-L{ggml_lib_dir}",
        "-lggml-base",
        *link_args,
    ]

    subprocess.check_call(cmd, cwd=repo_root)
    print(f"Built: {out}")
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ggml quants extension")
    parser.add_argument("--ggml-version", type=str, default="8027", help="The version of the ggml library to use")
    return parser.parse_args()

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
