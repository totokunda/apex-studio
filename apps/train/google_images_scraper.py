import argparse
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

import requests
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "images"


def build_search_url(query: str) -> str:
    return f"https://www.google.com/search?tbm=isch&hl=en&q={quote_plus(query)}"


def _click_first_matching(driver: webdriver.Chrome, xpaths: list[str]) -> bool:
    for xp in xpaths:
        try:
            el = driver.find_element(By.XPATH, xp)
            if el.is_displayed() and el.is_enabled():
                driver.execute_script("arguments[0].click();", el)
                return True
        except Exception:
            continue
    return False


def try_accept_consent(driver: webdriver.Chrome) -> None:
    # This varies by region/account; we best-effort click a consent button if present.
    xpaths = [
        "//*[@id='L2AGLb']",
        "//button[@id='L2AGLb']",
        "//button[.//div[normalize-space()='Accept all']]",
        "//button[normalize-space()='Accept all']",
        "//button[.//span[normalize-space()='Accept all']]",
        "//button[.//div[normalize-space()='I agree']]",
        "//button[normalize-space()='I agree']",
        "//button[.//span[normalize-space()='I agree']]",
        "//form//button[.//div[normalize-space()='Accept all']]",
    ]
    _click_first_matching(driver, xpaths)


@dataclass(frozen=True)
class ImageCandidate:
    url: str
    score: int


def extract_best_full_res_url(driver: webdriver.Chrome) -> Optional[str]:
    # Google Images uses multiple DOM variants; we collect all plausible "preview" images
    # and pick the largest (naturalWidth * naturalHeight).
    candidates: list[ImageCandidate] = []
    for css in ("img.n3VNCb", "img.sFlh5c", "img.iPVvYb"):
        try:
            imgs = driver.find_elements(By.CSS_SELECTOR, css)
        except Exception:
            imgs = []

        for img in imgs:
            try:
                src = (img.get_attribute("src") or "").strip()
                if not src.startswith("http"):
                    continue
                # Skip thumbnails and non-image payloads.
                if "gstatic.com/images?q=tbn" in src:
                    continue
                if src.startswith("https://encrypted-tbn0.gstatic.com/"):
                    continue
                if src.startswith("https://www.google.com/images/"):
                    continue

                w = driver.execute_script(
                    "return arguments[0].naturalWidth || 0;", img
                )
                h = driver.execute_script(
                    "return arguments[0].naturalHeight || 0;", img
                )
                score = int(w) * int(h)
                candidates.append(ImageCandidate(url=src, score=score))
            except Exception:
                continue

    if not candidates:
        return None
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0].url


def maybe_click_show_more(driver: webdriver.Chrome) -> None:
    # "Show more results" button at the bottom of the images grid.
    try:
        btn = driver.find_element(By.CSS_SELECTOR, "input.mye4qd")
        if btn.is_displayed() and btn.is_enabled():
            driver.execute_script("arguments[0].click();", btn)
    except Exception:
        return


def download_image(url: str, out_path: Path, timeout_s: int = 30) -> bool:
    try:
        with requests.get(
            url,
            headers={"User-Agent": UA},
            stream=True,
            timeout=timeout_s,
            allow_redirects=True,
        ) as r:
            if r.status_code != 200:
                return False
            ctype = (r.headers.get("Content-Type") or "").lower()
            if not ctype.startswith("image/"):
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def guess_extension(url: str, content_type: Optional[str] = None) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            "image/svg+xml": ".svg",
        }
        if ct in mapping:
            return mapping[ct]

    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".svg"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Scrape full-resolution image URLs from Google Images and optionally download them."
    )
    p.add_argument(
        "--query",
        default="sydney sweeney photos please",
        help="Google Images search query.",
    )
    p.add_argument("--limit", type=int, default=250, help="Max images to collect.")
    p.add_argument(
        "--output-urls",
        default="google_image_urls.txt",
        help="Where to write collected image URLs (one per line).",
    )
    p.add_argument(
        "--download-dir",
        default=None,
        help="If set, downloads images into this directory.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (recommended on servers).",
    )
    p.add_argument(
        "--max-scrolls",
        type=int,
        default=200,
        help="Safety cap for scroll attempts while loading results.",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=0.5,
        help="Minimum delay (seconds) between interactions.",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=1.2,
        help="Maximum delay (seconds) between interactions.",
    )
    args = p.parse_args()

    query = args.query
    limit = max(1, int(args.limit))
    output_urls = Path(args.output_urls)
    download_dir = Path(args.download_dir) if args.download_dir else None

    chrome_options = Options()
    if args.headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1400,1000")
    chrome_options.add_argument(f"--user-agent={UA}")

    # Allow custom Chrome binary if needed in CI.
    chrome_bin = os.environ.get("CHROME_BIN")
    if chrome_bin:
        chrome_options.binary_location = chrome_bin

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(build_search_url(query))
        time.sleep(1.0)
        try_accept_consent(driver)
        time.sleep(1.0)

        urls: list[str] = []
        url_set: set[str] = set()
        seen_thumb_keys: set[str] = set()

        output_urls.parent.mkdir(parents=True, exist_ok=True)
        if output_urls.exists():
            # Resume: load existing URLs to avoid redoing work.
            existing = {line.strip() for line in output_urls.read_text().splitlines() if line.strip()}
            url_set.update(existing)
            urls.extend(sorted(existing))

        scrolls = 0
        pbar = tqdm(total=limit, initial=len(url_set), desc="Collecting full-res URLs")
        while len(url_set) < limit and scrolls < args.max_scrolls:
            thumbs = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd, img.YQ4gaf")
            for thumb in thumbs:
                if len(url_set) >= limit:
                    break
                try:
                    thumb_key = (
                        thumb.get_attribute("src")
                        or thumb.get_attribute("data-src")
                        or thumb.get_attribute("alt")
                        or ""
                    ).strip()
                    if not thumb_key or thumb_key in seen_thumb_keys:
                        continue
                    seen_thumb_keys.add(thumb_key)

                    driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                        thumb,
                    )
                    driver.execute_script("arguments[0].click();", thumb)
                    time.sleep(random.uniform(args.min_delay, args.max_delay))

                    best = extract_best_full_res_url(driver)
                    if best and best not in url_set:
                        url_set.add(best)
                        urls.append(best)
                        with output_urls.open("a", encoding="utf-8") as f:
                            f.write(best + "\n")
                        pbar.update(1)
                except Exception:
                    continue

            # Scroll and try to load more results.
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(args.min_delay, args.max_delay) + 0.5)
            maybe_click_show_more(driver)
            time.sleep(random.uniform(args.min_delay, args.max_delay) + 0.5)
            scrolls += 1

        pbar.close()

        if download_dir:
            download_dir = download_dir / slugify(query)
            download_dir.mkdir(parents=True, exist_ok=True)

            ok = 0
            for i, url in tqdm(
                list(enumerate(urls[:limit])),
                total=min(limit, len(urls)),
                desc="Downloading images",
            ):
                # Use a stable-ish name to avoid collisions.
                host = urlparse(url).netloc.replace(":", "_")
                ext = guess_extension(url)
                out_path = download_dir / f"{i:04d}_{host}{ext}"
                if out_path.exists() and out_path.stat().st_size > 0:
                    ok += 1
                    continue
                if download_image(url, out_path=out_path):
                    ok += 1
                time.sleep(random.uniform(args.min_delay, args.max_delay))

            print(f"Downloaded {ok}/{min(limit, len(urls))} images to: {download_dir}")

        print(f"Wrote {min(limit, len(url_set))} URLs to: {output_urls}")
        return 0
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

