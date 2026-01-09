use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{thread_rng, Rng};
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_RANGE, RANGE};
use reqwest::Url;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use std::time::SystemTime;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tokio::time::sleep;

fn clamp_usize(n: usize, low: usize, high: usize) -> usize {
    if n < low {
        low
    } else if n > high {
        high
    } else {
        n
    }
}

fn pydict_to_headermap(headers: Option<&Bound<'_, PyDict>>) -> PyResult<HeaderMap> {
    let mut hm = HeaderMap::new();
    let Some(d) = headers else {
        return Ok(hm);
    };
    for (k, v) in d.iter() {
        let ks = k.extract::<String>()?;
        let vs = v.extract::<String>()?;
        let name = HeaderName::from_bytes(ks.as_bytes())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let value =
            HeaderValue::from_str(&vs).map_err(|e| PyValueError::new_err(e.to_string()))?;
        hm.insert(name, value);
    }
    Ok(hm)
}

fn ensure_parent_dir(path: &str) -> PyResult<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| PyOSError::new_err(e.to_string()))?;
        }
    }
    Ok(())
}

fn file_len_best_effort(path: &str) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn parse_total_from_content_range(cr: &str) -> Option<usize> {
    // Content-Range: bytes 0-0/702517648
    cr.split('/').last()?.trim().parse::<usize>().ok()
}

fn jitter_ms() -> usize {
    thread_rng().gen_range(0..=500)
}

fn default_ratelimit_wait() -> Duration {
    // Fallback when a server returns 429 without ratelimit/retry-after headers.
    // HuggingFace rate limits often require longer waits; use a more conservative default.
    Duration::from_millis((5000 + jitter_ms()) as u64)
}

fn exponential_backoff_ms(base_wait_ms: usize, n: usize, max_wait_ms: usize) -> usize {
    (base_wait_ms + n.pow(2) + jitter_ms()).min(max_wait_ms)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_millis() as u64
}

fn parse_range_line(line: &str) -> Option<(usize, usize)> {
    // "start,stop"
    let (a, b) = line.trim().split_once(',')?;
    let start = a.trim().parse::<usize>().ok()?;
    let stop = b.trim().parse::<usize>().ok()?;
    if stop >= start { Some((start, stop)) } else { None }
}

fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.is_empty() {
        return ranges;
    }
    ranges.sort_by_key(|(s, _)| *s);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    let mut cur = ranges[0];
    for (s, e) in ranges.into_iter().skip(1) {
        if s <= cur.1.saturating_add(1) {
            cur.1 = cur.1.max(e);
        } else {
            out.push(cur);
            cur = (s, e);
        }
    }
    out.push(cur);
    out
}

fn compute_missing_ranges(merged_done: &[(usize, usize)], length: usize) -> Vec<(usize, usize)> {
    // returns missing inclusive ranges within [0, length-1]
    if length == 0 {
        return vec![];
    }
    let mut missing = Vec::new();
    let mut cursor = 0usize;
    for (s, e) in merged_done.iter().copied() {
        if cursor < s {
            missing.push((cursor, s - 1));
        }
        cursor = cursor.max(e.saturating_add(1));
        if cursor >= length {
            break;
        }
    }
    if cursor < length {
        missing.push((cursor, length - 1));
    }
    missing
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct RateLimitInfo {
    resource_type: String,
    remaining: u64,
    reset_in_seconds: u64,
    limit: Option<u64>,
    window_seconds: Option<u64>,
}

fn parse_first_quoted_token(s: &str) -> Option<String> {
    let start = s.find('"')?;
    let rest = &s[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn parse_semicolon_kv_u64(s: &str, key: &str) -> Option<u64> {
    for part in s.split(';') {
        let part = part.trim();
        let Some((k, v)) = part.split_once('=') else {
            continue;
        };
        if k.trim() == key {
            return v.trim().parse::<u64>().ok();
        }
    }
    None
}

fn parse_ratelimit_headers(headers: &HeaderMap) -> Option<RateLimitInfo> {
    // Follows IETF draft (subset): https://www.ietf.org/archive/id/draft-ietf-httpapi-ratelimit-headers-09.html
    // Example:
    //   ratelimit: '"api";r=0;t=55'
    //   ratelimit-policy: '"fixed window";"api";q=500;w=300'
    let ratelimit = headers.get("ratelimit")?.to_str().ok()?;
    let resource_type = parse_first_quoted_token(ratelimit)?;
    let remaining = parse_semicolon_kv_u64(ratelimit, "r")?;
    let reset_in_seconds = parse_semicolon_kv_u64(ratelimit, "t")?;

    let mut limit: Option<u64> = None;
    let mut window_seconds: Option<u64> = None;
    if let Some(policy) = headers.get("ratelimit-policy").and_then(|v| v.to_str().ok()) {
        limit = parse_semicolon_kv_u64(policy, "q");
        window_seconds = parse_semicolon_kv_u64(policy, "w");
    }

    Some(RateLimitInfo {
        resource_type,
        remaining,
        reset_in_seconds,
        limit,
        window_seconds,
    })
}

fn parse_retry_after_seconds(headers: &HeaderMap) -> Option<u64> {
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
}

enum DownloadChunkError {
    RateLimited(Duration),
    Other(PyErr),
}

impl From<PyErr> for DownloadChunkError {
    fn from(value: PyErr) -> Self {
        Self::Other(value)
    }
}

#[pyfunction]
#[pyo3(signature = (
    url,
    file_path,
    part_path,
    headers=None,
    verify_tls=true,
    progress_callback=None,
    adaptive=true,
    chunk_size=1024*1024,
    initial_chunk_size=512*1024,
    target_chunk_seconds=0.25,
    min_chunk_size=64*1024,
    max_chunk_size=16*1024*1024,
    callback_min_interval_secs=0.2,
    callback_min_bytes=1024*1024
))]
fn download_from_url(
    py: Python<'_>,
    url: String,
    file_path: String,
    part_path: String,
    headers: Option<&Bound<'_, PyDict>>,
    verify_tls: bool,
    progress_callback: Option<PyObject>,
    adaptive: bool,
    chunk_size: usize,
    initial_chunk_size: usize,
    target_chunk_seconds: f64,
    min_chunk_size: usize,
    max_chunk_size: usize,
    callback_min_interval_secs: f64,
    callback_min_bytes: usize,
) -> PyResult<String> {
    // If already exists, skip
    if Path::new(&file_path).exists() {
        return Ok(file_path);
    }
    ensure_parent_dir(&file_path)?;

    let mut hm = pydict_to_headermap(headers)?;
    // Prefer identity encoding so content-length matches bytes written in most cases.
    if !hm.contains_key("accept-encoding") {
        hm.insert(
            HeaderName::from_static("accept-encoding"),
            HeaderValue::from_static("identity"),
        );
    }

    // Resume size from .part
    let resume_size = if Path::new(&part_path).exists() {
        file_len_best_effort(&part_path)
    } else {
        0
    };

    // Release GIL for the download; reacquire only for throttled callback calls.
    let url2 = url.clone();
    let file_path2 = file_path.clone();
    let part_path2 = part_path.clone();
    let cb2 = progress_callback;
    let hm2 = hm.clone();
    let name2 = Path::new(&file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string());

    // Parallelization controls (hf_transfer-style) via env vars (signature unchanged):
    // - APEX_RUST_DOWNLOAD_MAX_FILES (default 8)
    // - APEX_RUST_DOWNLOAD_PARALLEL_FAILURES (default 0)
    // - APEX_RUST_DOWNLOAD_MAX_RETRIES (default 0)
    let max_files = env_usize("APEX_RUST_DOWNLOAD_MAX_FILES", 8).max(1);
    let parallel_failures = env_usize("APEX_RUST_DOWNLOAD_PARALLEL_FAILURES", 3);
    let max_retries = env_usize("APEX_RUST_DOWNLOAD_MAX_RETRIES", 5);
    let base_wait_ms = env_usize("APEX_RUST_DOWNLOAD_RETRY_BASE_MS", 300);
    let max_wait_ms = env_usize("APEX_RUST_DOWNLOAD_RETRY_MAX_MS", 10_000);
    let target_chunk_seconds = target_chunk_seconds.max(0.0);
    let adaptive_min_chunk_size = env_usize(
        "APEX_RUST_DOWNLOAD_ADAPTIVE_MIN_CHUNK_SIZE",
        min_chunk_size.max(1024 * 1024), // avoid tiny ranges that can crater throughput
    );
    // Initial chunk size (adaptive) or fixed chunk size (non-adaptive)
    let init_chunk_size = if adaptive {
        clamp_usize(
            initial_chunk_size.max(adaptive_min_chunk_size),
            adaptive_min_chunk_size,
            max_chunk_size,
        )
    } else {
        clamp_usize(chunk_size, min_chunk_size, max_chunk_size)
    };

    py.allow_threads(move || -> PyResult<()> {
        // Build a tokio runtime (hf_transfer-style)
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        rt.block_on(async move {
            // Build async client (http2 keepalive helps for many range requests)
            let mut client_builder = reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .http2_keep_alive_timeout(Duration::from_secs(15))
                .http2_adaptive_window(true)
                .tcp_keepalive(Some(Duration::from_secs(60)))
                .tcp_nodelay(true)
                .pool_max_idle_per_host(max_files)
                .redirect(reqwest::redirect::Policy::limited(10));
            if !verify_tls {
                client_builder = client_builder.danger_accept_invalid_certs(true);
            }
            let client = client_builder
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Handle redirected URL only once (hf_transfer trick):
            // - make a single 0-0 range request to obtain final URL + content length from Content-Range
            // - then download all chunks from the redirected URL to avoid extra hub requests
            let mut headers = hm2.clone();
            let auth_header = headers.remove(AUTHORIZATION);

            let initial = loop {
                let rb = if let Some(token) = auth_header.as_ref() {
                    client.get(&url2).header(AUTHORIZATION, token)
                } else {
                    client.get(&url2)
                }
                .headers(headers.clone())
                .header(RANGE, "bytes=0-0");

                let resp = rb
                    .send()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Error while downloading: {e}")))?;

                if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE
                {
                    // Never fail the overall request due to 429/503; wait and retry.
                    // Prefer standardized ratelimit headers, fall back to Retry-After,
                    // then fall back to a small default backoff.
                    if let Some(info) = parse_ratelimit_headers(resp.headers()) {
                        sleep(Duration::from_secs(info.reset_in_seconds.max(1))).await;
                    } else if let Some(secs) = parse_retry_after_seconds(resp.headers()) {
                        sleep(Duration::from_secs(secs.max(1))).await;
                    } else {
                        sleep(default_ratelimit_wait()).await;
                    }
                    continue;
                }

                break resp
                    .error_for_status()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            };

            let redirected_url = initial.url().to_string();

            // Re-add Authorization header only if redirect host matches original host
            if let (Ok(orig), Ok(redir)) = (Url::parse(&url2), Url::parse(&redirected_url)) {
                if orig.host_str() == redir.host_str() {
                    if let Some(token) = auth_header {
                        headers.insert(AUTHORIZATION, token);
                    }
                }
            }

            let cr = initial
                .headers()
                .get(CONTENT_RANGE)
                .ok_or_else(|| PyRuntimeError::new_err("No content length (missing Content-Range)"))?
                .to_str()
                .map_err(|e| PyRuntimeError::new_err(format!("Error while downloading: {e}")))?;

            let length = parse_total_from_content_range(cr)
                .ok_or_else(|| PyRuntimeError::new_err("Error while downloading: could not parse size"))?;

            // Resume bookkeeping: because we preallocate the .part file, its length alone is NOT a safe
            // indicator of completion. We track completed ranges in a sidecar file.
            let ranges_path = format!("{part_path2}.ranges");
            let part_exists = Path::new(&part_path2).exists();
            let ranges_exists = Path::new(&ranges_path).exists();

            let mut completed: Vec<(usize, usize)> = Vec::new();
            if part_exists {
                if !ranges_exists {
                    // Unsafe to resume without metadata; start clean.
                    let _ = fs::remove_file(&part_path2);
                } else {
                    // Parse completed ranges (best-effort).
                    if let Ok(txt) = fs::read_to_string(&ranges_path) {
                        for line in txt.lines() {
                            if let Some((s, e)) = parse_range_line(line) {
                                completed.push((s, e));
                            }
                        }
                    }
                }
            }
            let completed = merge_ranges(completed);
            if !completed.is_empty() {
                // If all bytes covered, finalize quickly.
                if completed.len() == 1 && completed[0].0 == 0 && completed[0].1 + 1 >= length {
                    ensure_parent_dir(&file_path2)?;
                    fs::rename(&part_path2, &file_path2)
                        .map_err(|e| PyOSError::new_err(e.to_string()))?;
                    let _ = fs::remove_file(&ranges_path);
                    if let Some(cb) = cb2.as_ref() {
                        Python::with_gil(|py| {
                            let cb = cb.bind(py);
                            cb.call1((length as i64, Some(length as i64), name2.clone()))?;
                            Ok::<(), PyErr>(())
                        })?;
                    }
                    return Ok::<(), PyErr>(());
                }
            }

            // Pre-create the .part file and set its length so random writes succeed
            ensure_parent_dir(&part_path2)?;
            {
                let f = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(&part_path2)
                    .await
                    .map_err(|e| PyOSError::new_err(e.to_string()))?;
                f.set_len(length as u64)
                    .await
                    .map_err(|e| PyOSError::new_err(e.to_string()))?;
            }

            // Progress tracking: downloaded bytes (contiguous resume bytes count as already done)
            let done_total: u64 = completed
                .iter()
                .map(|(s, e)| (*e as u64).saturating_sub(*s as u64).saturating_add(1))
                .sum();
            let downloaded = Arc::new(AtomicU64::new(done_total.max(resume_size)));
            let total_len_u64 = length as u64;
            let cb_interval = Duration::from_secs_f64(callback_min_interval_secs.max(0.0));
            let cb_min_bytes = callback_min_bytes.max(1) as u64;
            let mut last_cb_at = Instant::now();
            let mut last_cb_bytes = downloaded.load(Ordering::Relaxed);

            // Work scheduling: queue missing ranges, split into chunks dynamically (adaptive) or fixed.
            let missing = if completed.is_empty() {
                vec![(0usize, length - 1)]
            } else {
                compute_missing_ranges(&completed, length)
            };
            let work = Arc::new(Mutex::new(VecDeque::<(usize, usize)>::from(missing)));

            // Adaptive state shared across workers
            #[derive(Debug)]
            struct AdaptiveState {
                chunk_size: usize,
                speed_bps: Option<f64>,
            }
            let adaptive_state = Arc::new(Mutex::new(AdaptiveState {
                chunk_size: init_chunk_size,
                speed_bps: None,
            }));

            // Serialize appends to the ranges sidecar file
            let ranges_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&ranges_path)
                .await
                .map_err(|e| PyOSError::new_err(e.to_string()))?;
            let ranges_file = Arc::new(Mutex::new(ranges_file));

            // Failure controls/cancellation
            let cancel = Arc::new(AtomicBool::new(false));
            let parallel_failures_semaphore = Arc::new(tokio::sync::Semaphore::new(parallel_failures.max(1)));
            // Global pause for all workers when any worker hits a 429/503 (prevents thrash)
            let rate_limited_until_ms = Arc::new(AtomicU64::new(0));

            async fn download_chunk(
                client: &reqwest::Client,
                url: &str,
                file: &mut tokio::fs::File,
                start: usize,
                stop: usize,
                headers: HeaderMap,
            ) -> Result<usize, DownloadChunkError> {
                let range = format!("bytes={start}-{stop}");
                file.seek(std::io::SeekFrom::Start(start as u64))
                    .await
                    .map_err(|e| PyOSError::new_err(e.to_string()))?;
                let resp = client
                    .get(url)
                    .headers(headers)
                    .header(RANGE, range)
                    .send()
                    .await
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE
                {
                    // Never treat 429/503 as a hard error; report a wait duration.
                    if let Some(info) = parse_ratelimit_headers(resp.headers()) {
                        return Err(DownloadChunkError::RateLimited(Duration::from_secs(
                            info.reset_in_seconds.max(1),
                        )));
                    } else if let Some(secs) = parse_retry_after_seconds(resp.headers()) {
                        return Err(DownloadChunkError::RateLimited(Duration::from_secs(secs.max(1))));
                    } else {
                        return Err(DownloadChunkError::RateLimited(default_ratelimit_wait()));
                    }
                }

                let resp = resp
                    .error_for_status()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let mut wrote = 0usize;
                let mut stream = resp.bytes_stream();
                while let Some(item) = stream.next().await {
                    let bytes = item.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    if bytes.is_empty() {
                        continue;
                    }
                    file.write_all(&bytes)
                        .await
                        .map_err(|e| PyOSError::new_err(e.to_string()))?;
                    wrote += bytes.len();
                }
                Ok(wrote)
            }

            // Spawn a fixed-size worker pool (significantly less overhead than spawning per-chunk)
            let mut handles = Vec::with_capacity(max_files);
            for _ in 0..max_files {
                let client = client.clone();
                let url = redirected_url.clone();
                let headers = headers.clone();
                let work = work.clone();
                let downloaded = downloaded.clone();
                let adaptive_state = adaptive_state.clone();
                let ranges_file = ranges_file.clone();
                let cancel = cancel.clone();
                let pf_sem = parallel_failures_semaphore.clone();
                let part_path_worker = part_path2.clone();
                let rate_limited_until_ms = rate_limited_until_ms.clone();

                handles.push(tokio::spawn(async move {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .truncate(false)
                        .create(true)
                        .open(&part_path_worker)
                        .await
                        .map_err(|e| PyOSError::new_err(e.to_string()))?;

                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        // Respect global rate-limit pause (if any)
                        loop {
                            let until = rate_limited_until_ms.load(Ordering::Relaxed);
                            if until == 0 {
                                break;
                            }
                            let now = now_millis();
                            if now >= until {
                                // best-effort clear
                                rate_limited_until_ms.store(0, Ordering::Relaxed);
                                break;
                            }
                            sleep(Duration::from_millis((until - now).min(30_000))).await;
                            if cancel.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                        // determine current piece size
                        let piece_size = if adaptive {
                            let st = adaptive_state.lock().await;
                            st.chunk_size
                        } else {
                            init_chunk_size
                        };

                        let maybe_piece = {
                            let mut q = work.lock().await;
                            let (s, e) = match q.pop_front() {
                                None => break,
                                Some(v) => v,
                            };
                            let max_stop = s.saturating_add(piece_size.saturating_sub(1));
                            if max_stop >= e {
                                Some((s, e))
                            } else {
                                // split: take [s, max_stop], push remainder back to front
                                let rem = (max_stop + 1, e);
                                q.push_front(rem);
                                Some((s, max_stop))
                            }
                        };
                        let Some((start, stop)) = maybe_piece else {
                            break;
                        };

                        let mut attempt = 0usize;
                        loop {
                            if cancel.load(Ordering::Relaxed) {
                                break;
                            }
                            let t0 = Instant::now();
                            match download_chunk(&client, &url, &mut file, start, stop, headers.clone()).await {
                                Ok(wrote) => {
                                    let dt = t0.elapsed().as_secs_f64();
                                    if adaptive && dt > 0.0 && wrote > 0 {
                                        let inst_bps = (wrote as f64) / dt;
                                        let mut st = adaptive_state.lock().await;
                                        st.speed_bps = Some(match st.speed_bps {
                                            None => inst_bps,
                                            Some(prev) => 0.7 * prev + 0.3 * inst_bps,
                                        });
                                        if let Some(speed) = st.speed_bps {
                                            let desired = (speed * target_chunk_seconds.max(0.05)) as usize;
                                            // Prevent huge swings in chunk size from transient stalls:
                                            // allow at most 2x up or 0.5x down per update.
                                            let cur = st.chunk_size.max(adaptive_min_chunk_size);
                                            let lower = (cur as f64 * 0.5) as usize;
                                            let upper = (cur as f64 * 2.0) as usize;
                                            let bounded = desired.clamp(lower.max(adaptive_min_chunk_size), upper.max(adaptive_min_chunk_size));
                                            st.chunk_size = clamp_usize(bounded, adaptive_min_chunk_size, max_chunk_size);
                                        }
                                    }
                                    downloaded.fetch_add(wrote as u64, Ordering::Relaxed);
                                    // record completed range for safe resume
                                    {
                                        let mut rf = ranges_file.lock().await;
                                        rf.write_all(format!("{start},{stop}\n").as_bytes())
                                            .await
                                            .map_err(|e| PyOSError::new_err(e.to_string()))?;
                                    }
                                    break;
                                }
                                Err(DownloadChunkError::RateLimited(wait)) => {
                                    // Pause all workers and increase chunk size to reduce request rate.
                                    let until = now_millis().saturating_add(wait.as_millis() as u64);
                                    rate_limited_until_ms.fetch_max(until, Ordering::Relaxed);
                                    if adaptive {
                                        let mut st = adaptive_state.lock().await;
                                        st.chunk_size = st.chunk_size.max(max_chunk_size);
                                    }
                                    sleep(wait).await;
                                }
                                Err(DownloadChunkError::Other(e)) => {
                                    if parallel_failures == 0 || max_retries == 0 {
                                        cancel.store(true, Ordering::Relaxed);
                                        return Err(e);
                                    }
                                    if attempt >= max_retries {
                                        cancel.store(true, Ordering::Relaxed);
                                        return Err(PyRuntimeError::new_err(format!(
                                            "Failed after too many retries ({max_retries})"
                                        )));
                                    }
                                    // Limit failures in parallel (avoid self-DOS)
                                    let _pf_permit = pf_sem
                                        .clone()
                                        .try_acquire_owned()
                                        .map_err(|e| PyRuntimeError::new_err(format!(
                                            "Failed too many failures in parallel ({parallel_failures}): {e}"
                                        )))?;
                                    let wait_ms = exponential_backoff_ms(base_wait_ms, attempt, max_wait_ms);
                                    sleep(Duration::from_millis(wait_ms as u64)).await;
                                    attempt += 1;
                                }
                            }
                        }
                    }
                    Ok::<(), PyErr>(())
                }));
            }

            // Drive progress callback while workers run; fail fast on worker errors
            let mut joins = futures::stream::FuturesUnordered::new();
            for h in handles {
                joins.push(h);
            }
            loop {
                tokio::select! {
                    res = joins.next() => {
                        let Some(res) = res else { break; };
                        match res {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                cancel.store(true, Ordering::Relaxed);
                                let _ = fs::remove_file(&part_path2);
                                let _ = fs::remove_file(&ranges_path);
                                return Err(e);
                            }
                            Err(e) => {
                                cancel.store(true, Ordering::Relaxed);
                                let _ = fs::remove_file(&part_path2);
                                let _ = fs::remove_file(&ranges_path);
                                return Err(PyRuntimeError::new_err(format!("Error while downloading: {e}")));
                            }
                        }
                    }
                    _ = sleep(Duration::from_millis(100)) => {
                        // tick
                    }
                }
                if cb2.is_some() {
                    let now = Instant::now();
                    let n = downloaded.load(Ordering::Relaxed);
                    if now.duration_since(last_cb_at) >= cb_interval
                        || n.saturating_sub(last_cb_bytes) >= cb_min_bytes
                    {
                        last_cb_at = now;
                        last_cb_bytes = n;
                        Python::with_gil(|py| -> PyResult<()> {
                            if let Some(cb) = cb2.as_ref() {
                                cb.call1(py, (n as i64, Some(total_len_u64 as i64), name2.clone()))?;
                            }
                            Ok(())
                        })?;
                    }
                }
            }

            // Mark complete & final callback
            downloaded.store(total_len_u64, Ordering::Relaxed);
            if let Some(cb) = cb2.as_ref() {
                Python::with_gil(|py| {
                    cb.call1(py, (total_len_u64 as i64, Some(total_len_u64 as i64), name2.clone()))?;
                    Ok::<(), PyErr>(())
                })?;
            }

            // Atomic finalize: rename .part -> final
            ensure_parent_dir(&file_path2)?;
            fs::rename(&part_path2, &file_path2).map_err(|e| PyOSError::new_err(e.to_string()))?;
            let _ = fs::remove_file(&ranges_path);
            Ok::<(), PyErr>(())
        })?;
        Ok(())
    })?;

    Ok(file_path)
}

#[pymodule]
fn apex_download_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(download_from_url, m)?)?;
    Ok(())
}


