#!/usr/bin/env python3
"""
Test script for /components/download and websocket progress.

Usage examples:
  python scripts/test_components_download.py \
    --url http://127.0.0.1:8765 \
    --paths https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-000001.safetensors,
            https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore

Notes:
  - Ensure the backend server is running.
  - If 'websocket-client' is installed, the script will attach to /ws/job/{job_id} for live updates.
    Otherwise it will fall back to HTTP polling.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from typing import List, Optional

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test components download with progress"
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8765", help="Backend base URL"
    )
    parser.add_argument(
        "--paths",
        default="https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/model-00001-of-000001.safetensors",
        help="Comma-separated list of paths to download",
    )
    parser.add_argument(
        "--save-path", default=None, help="Optional save path on server"
    )
    parser.add_argument(
        "--no-ws", action="store_true", help="Disable websocket; poll status only"
    )
    return parser.parse_args()


def as_ws_url(http_url: str) -> str:
    if http_url.startswith("https://"):
        return "wss://" + http_url[len("https://") :]
    if http_url.startswith("http://"):
        return "ws://" + http_url[len("http://") :]
    # Default to ws:// if protocol missing
    return "ws://" + http_url


def start_download(
    base_url: str, paths: List[str], save_path: Optional[str] = None
) -> str:
    payload = {"paths": paths}
    if save_path:
        payload["save_path"] = save_path
    resp = requests.post(f"{base_url}/components/download", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("job_id") or data.get("data", {}).get("job_id")
    if not job_id:
        raise RuntimeError(f"Unexpected response: {data}")
    return job_id


def poll_status_until_done(
    base_url: str, job_id: str, stop_event: threading.Event
) -> str:
    status = "unknown"
    while not stop_event.is_set():
        try:
            r = requests.get(f"{base_url}/components/status/{job_id}", timeout=10)
            if r.ok:
                data = r.json()
                status = data.get("status", status)
                if status in {"complete", "error", "cancelled"}:
                    return status
        except Exception:
            pass
        time.sleep(1.0)
    return status


def run_ws_loop(ws_url: str, job_id: str, stop_event: threading.Event) -> None:
    try:
        from websocket import WebSocketApp  # type: ignore
    except Exception:
        print(
            "websocket-client not installed; skipping WS (pip install websocket-client)"
        )
        return

    def on_message(_ws, message: str):
        try:
            msg = json.loads(message)
            progress = msg.get("progress")
            pct = (
                f"{progress * 100:.1f}%"
                if isinstance(progress, (int, float))
                else "n/a"
            )
            label = msg.get("metadata", {}).get("label")
            status = msg.get("status", "processing")
            line = f"progress={pct} status={status}"
            if label:
                line += f" [{label}]"
            if msg.get("message"):
                line += f" {msg['message']}"
            print(line)
        except Exception:
            print(message)

    def on_error(_ws, error):
        print(f"WS error: {error}")

    def on_close(_ws, _code, _msg):
        # Allow main thread to finish
        pass

    app = WebSocketApp(
        f"{ws_url}/ws/job/{job_id}",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    def _run():
        try:
            app.run_forever()
        except Exception as e:
            print(f"WS run error: {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Stop when main signals done
    while not stop_event.is_set():
        time.sleep(0.25)
    try:
        app.close()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    base_url = args.url.rstrip("/")
    paths = ["deepseek-ai/DeepSeek-OCR/model-00001-of-000001.safetensors"]
    if not paths:
        print("No paths provided")
        return 2

    print(f"Backend: {base_url}")
    print("Paths:")
    for p in paths:
        print(f"  - {p}")

    try:
        job_id = start_download(base_url, paths, args.save_path)
    except Exception as e:
        print(f"Failed to start download: {e}")
        return 1

    print(f"Started job: {job_id}")

    stop_event = threading.Event()

    if not args.no_ws:
        ws_url = as_ws_url(base_url)
        run_thread = threading.Thread(
            target=run_ws_loop, args=(ws_url, job_id, stop_event), daemon=True
        )
        run_thread.start()

    final = poll_status_until_done(base_url, job_id, stop_event)
    stop_event.set()
    print(f"Final status: {final}")
    return 0 if final == "complete" else 1 if final == "error" else 2


if __name__ == "__main__":
    sys.exit(main())
