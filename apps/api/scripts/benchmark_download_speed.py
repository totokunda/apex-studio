"""
Benchmark Python vs Rust URL download throughput with tqdm progress callbacks.

Usage:
  python scripts/benchmark_download_speed.py \
    --url "https://huggingface.co/lodestones/Chroma1-HD/resolve/main/transformer/diffusion_pytorch_model-00002-of-00002.safetensors" \
    --out-dir /tmp/apex_download_bench

Notes:
- The Rust path requires the optional module `apex_download_rs` to be installed:
    cd rust/apex_download_rs && maturin develop --release
- Both implementations use the SAME progress_callback (tqdm) so callback overhead is comparable.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import requests
from tqdm import tqdm

from src.utils.defaults import DEFAULT_HEADERS


ProgressCb = Callable[[int, Optional[int], Optional[str]], None]


def _requests_verify() -> bool:
    """Mirror DownloadMixin._requests_verify()."""
    try:
        flag = os.environ.get("APEX_REQUESTS_VERIFY")
        if flag is None:
            return True
        return str(flag).strip().lower() not in ("0", "false", "no", "off")
    except Exception:
        return True


def _mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _human_mb_per_s(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (num_bytes / (1024 * 1024)) / seconds


def _progress_tqdm(desc: str) -> Tuple[tqdm, ProgressCb]:
    """
    Create a tqdm bar and a progress callback matching DownloadMixin signature:
      cb(downloaded_so_far, total_or_none, filename_or_none)
    """
    bar = tqdm(
        desc=desc,
        total=None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )
    state = {"last_n": 0, "total": None}

    def cb(n: int, total: Optional[int], _label: Optional[str] = None) -> None:
        try:
            if total is not None and state["total"] != total:
                bar.total = int(total)
                state["total"] = int(total)
            n_int = int(n or 0)
            delta = n_int - int(state["last_n"])
            if delta > 0:
                bar.update(delta)
            state["last_n"] = n_int
        except Exception:
            pass

    return bar, cb


def download_python_requests(
    *,
    url: str,
    dest_path: str,
    progress_callback: Optional[ProgressCb],
    chunk_size: int = 1024 * 1024,
    timeout: int = 30,
) -> int:
    """
    Pure-Python baseline using requests streaming.
    Returns bytes written.
    """
    _mkdirp(os.path.dirname(dest_path) or ".")
    headers = dict(DEFAULT_HEADERS)
    headers.setdefault("Accept-Encoding", "identity")

    # HEAD for total (best-effort)
    total: Optional[int] = None
    try:
        head = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers=headers,
            verify=_requests_verify(),
        )
        if head.ok:
            try:
                total = int(head.headers.get("content-length", "0")) or None
            except Exception:
                total = None
    except Exception:
        total = None

    written = 0
    with requests.get(
        url,
        allow_redirects=True,
        timeout=timeout,
        headers=headers,
        stream=True,
        verify=_requests_verify(),
    ) as r:
        r.raise_for_status()
        if total is None:
            try:
                cl = int(r.headers.get("content-length", "0")) or None
                total = cl
            except Exception:
                total = None

        # Ensure content is decoded consistently
        try:
            r.raw.decode_content = True
        except Exception:
            pass

        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if progress_callback:
                    progress_callback(written, total, os.path.basename(dest_path))

    return written


def download_rust_extension(
    *,
    url: str,
    dest_path: str,
    progress_callback: Optional[ProgressCb],
    adaptive: bool = True,
    initial_chunk_size: int = 512 * 1024,
    target_chunk_seconds: float = 0.25,
    min_chunk_size: int = 64 * 1024,
    max_chunk_size: int = 16 * 1024 * 1024,
    chunk_size: int = 1024 * 1024,
    callback_min_interval_secs: float = 0.2,
    callback_min_bytes: int = 1024 * 1024,
) -> int:
    """
    Rust implementation via `apex_download_rs` (must be installed).
    Returns bytes written.
    """
    from apex_download_rs import download_from_url as rs_download_from_url  # type: ignore

    _mkdirp(os.path.dirname(dest_path) or ".")
    part_path = f"{dest_path}.part"
    _remove_if_exists(dest_path)
    _remove_if_exists(part_path)

    headers = dict(DEFAULT_HEADERS)
    headers.setdefault("Accept-Encoding", "identity")

    rs_download_from_url(
        url=url,
        file_path=dest_path,
        part_path=part_path,
        headers=headers,
        verify_tls=_requests_verify(),
        progress_callback=progress_callback,
        adaptive=adaptive,
        chunk_size=int(chunk_size),
        initial_chunk_size=int(initial_chunk_size),
        target_chunk_seconds=float(target_chunk_seconds),
        min_chunk_size=int(min_chunk_size),
        max_chunk_size=int(max_chunk_size),
        callback_min_interval_secs=float(callback_min_interval_secs),
        callback_min_bytes=int(callback_min_bytes),
    )

    try:
        return os.path.getsize(dest_path)
    except Exception:
        return 0


@dataclass
class Result:
    name: str
    seconds: float
    bytes_written: int

    @property
    def mb_per_s(self) -> float:
        return _human_mb_per_s(self.bytes_written, self.seconds)


def _bench_one(name: str, fn: Callable[[], int]) -> Result:
    t0 = time.perf_counter()
    n = fn()
    dt = time.perf_counter() - t0
    return Result(name=name, seconds=dt, bytes_written=int(n or 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out-dir", default="/tmp/apex_download_bench")
    ap.add_argument("--python-chunk-bytes", type=int, default=1024 * 1024)
    ap.add_argument("--rust", action="store_true", help="Run Rust benchmark (requires apex_download_rs)")
    ap.add_argument("--python", action="store_true", help="Run Python benchmark")
    args = ap.parse_args()

    _mkdirp(args.out_dir)
    url = args.url

    # Default: run both (if rust is installed)
    run_python = args.python or (not args.python and not args.rust)
    run_rust = args.rust or (not args.python and not args.rust)

    results: list[Result] = []

    if run_python:
        py_dest = os.path.join(args.out_dir, "python_download.bin")
        _remove_if_exists(py_dest)
        bar, cb = _progress_tqdm("python")
        try:
            results.append(
                _bench_one(
                    "python",
                    lambda: download_python_requests(
                        url=url,
                        dest_path=py_dest,
                        progress_callback=cb,
                        chunk_size=args.python_chunk_bytes,
                    ),
                )
            )
        finally:
            bar.close()

    if run_rust:
        rust_dest = os.path.join(args.out_dir, "rust_download.bin")
        try:
            import apex_download_rs  # noqa: F401
        except Exception as e:
            print(
                "Rust benchmark skipped: `apex_download_rs` not installed.\n"
                "Build it with:\n"
                "  cd rust/apex_download_rs && maturin develop --release\n"
                f"Import error: {e}"
            )
        else:
            bar, cb = _progress_tqdm("rust")
            try:
                results.append(
                    _bench_one(
                        "rust",
                        lambda: download_rust_extension(
                            url=url,
                            dest_path=rust_dest,
                            progress_callback=cb,
                        ),
                    )
                )
            finally:
                bar.close()

    print("\n=== Results ===")
    for r in results:
        print(
            f"{r.name:>6}: {r.seconds:8.2f}s  {r.bytes_written/(1024*1024):10.2f} MiB  {r.mb_per_s:8.2f} MiB/s"
        )


if __name__ == "__main__":
    main()


