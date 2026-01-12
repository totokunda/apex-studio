"""
Minimal diagnostic for Hugging Face Hub "resolve -> signed CDN URL" downloads.

Why this exists:
- In some environments (corporate proxies / DPI), HF CDN downloads can be slow or flaky.
- A short read timeout can make signed URLs look "broken" even though they work.

Usage:
  python scripts/dev/hf_cdn_diagnostic.py --repo gpt2 --file pytorch_model.bin
  python scripts/dev/hf_cdn_diagnostic.py --repo bert-base-uncased --file config.json
"""

from __future__ import annotations

import argparse
import time
from urllib.parse import urlparse

import requests
from huggingface_hub import hf_hub_url


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="HF repo id, e.g. gpt2")
    p.add_argument("--file", required=True, help="Filename in repo, e.g. pytorch_model.bin")
    p.add_argument("--rev", default="main", help="Revision, default: main")
    p.add_argument("--bytes", type=int, default=1024, help="How many bytes to fetch via Range")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--read-timeout", type=float, default=180.0)
    p.add_argument(
        "--no-proxy-env",
        action="store_true",
        help="Disable requests' use of proxy env vars (sets Session.trust_env=False)",
    )
    args = p.parse_args()

    resolve_url = hf_hub_url(repo_id=args.repo, filename=args.file, revision=args.rev)
    print("resolve_url:", resolve_url)

    timeout = (args.connect_timeout, args.read_timeout)

    # Resolve to signed URL (if any) without following redirects.
    r = requests.head(resolve_url, allow_redirects=False, timeout=timeout)
    print("\nHEAD resolve (allow_redirects=False)")
    print("  status:", r.status_code)
    signed_url = r.headers.get("location") or resolve_url
    print("  location:", r.headers.get("location"))
    print("  chosen_url:", signed_url)
    print("  host:", urlparse(signed_url).netloc)

    # Fetch a small Range to validate body bytes can be read.
    sess = requests.Session()
    sess.trust_env = not args.no_proxy_env
    sess.headers.setdefault("User-Agent", "apex-studio-hf-cdn-diagnostic/1.0")
    sess.headers.setdefault("Accept-Encoding", "identity")

    n = max(1, int(args.bytes))
    headers = {"Range": f"bytes=0-{n-1}"}
    print(f"\nGET Range 0-{n-1} (trust_env={sess.trust_env})")
    t0 = time.time()
    resp = sess.get(signed_url, stream=True, allow_redirects=True, timeout=timeout, headers=headers)
    print("  status:", resp.status_code)
    print("  final_url:", resp.url)
    print("  content-range:", resp.headers.get("content-range"))
    print("  content-length:", resp.headers.get("content-length"))
    try:
        buf = b""
        for chunk in resp.iter_content(chunk_size=min(64 * 1024, n)):
            if not chunk:
                continue
            buf += chunk
            if len(buf) >= n:
                buf = buf[:n]
                break
        dt = time.time() - t0
        print("  read_len:", len(buf))
        print("  time_s:", round(dt, 3))
        return 0 if len(buf) == n else 2
    finally:
        resp.close()


if __name__ == "__main__":
    raise SystemExit(main())

