use futures::stream::FuturesUnordered;
use futures::StreamExt;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{thread_rng, Rng};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_RANGE, RANGE};
use reqwest::Url;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Semaphore;
use tokio::time::sleep;

const BASE_WAIT_TIME_MS: usize = 300;
const MAX_WAIT_TIME_MS: usize = 10_000;

fn jitter_ms() -> usize {
    thread_rng().gen_range(0..=500)
}

fn exponential_backoff_ms(base: usize, attempt: usize, max: usize) -> usize {
    (base + attempt.pow(2) * 100 + jitter_ms()).min(max)
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

fn parse_total_from_content_range(cr: &str) -> Option<usize> {
    cr.split('/').last()?.trim().parse::<usize>().ok()
}

fn default_ratelimit_wait() -> Duration {
    Duration::from_millis((5000 + jitter_ms()) as u64)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_range_line(line: &str) -> Option<(usize, usize)> {
    let (a, b) = line.trim().split_once(',')?;
    let start = a.trim().parse::<usize>().ok()?;
    let stop = b.trim().parse::<usize>().ok()?;
    if stop >= start {
        Some((start, stop))
    } else {
        None
    }
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

fn parse_ratelimit_reset(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    // Try standard headers
    if let Some(v) = headers.get("retry-after").and_then(|v| v.to_str().ok()) {
        if let Ok(secs) = v.trim().parse::<u64>() {
            return Some(secs.max(1));
        }
    }
    // Try ratelimit header (IETF draft)
    if let Some(v) = headers.get("ratelimit").and_then(|v| v.to_str().ok()) {
        for part in v.split(';') {
            let part = part.trim();
            if let Some((k, val)) = part.split_once('=') {
                if k.trim() == "t" {
                    if let Ok(secs) = val.trim().parse::<u64>() {
                        return Some(secs.max(1));
                    }
                }
            }
        }
    }
    None
}

#[derive(Debug)]
enum ChunkError {
    RateLimited(Duration),
    Retryable(String),
    Fatal(String),
}

/// Download a single chunk to memory, then write to file at offset.
/// This matches hf_transfer's approach for maximum throughput.
async fn download_chunk(
    client: &reqwest::Client,
    url: &str,
    filename: &str,
    start: usize,
    stop: usize,
    headers: HeaderMap,
) -> Result<usize, ChunkError> {
    let range = format!("bytes={start}-{stop}");

    let response = client
        .get(url)
        .headers(headers)
        .header(RANGE, range)
        .send()
        .await
        .map_err(|e| ChunkError::Retryable(e.to_string()))?;

    let status = response.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
    {
        let wait = parse_ratelimit_reset(response.headers())
            .map(Duration::from_secs)
            .unwrap_or_else(default_ratelimit_wait);
        return Err(ChunkError::RateLimited(wait));
    }

    let response = response
        .error_for_status()
        .map_err(|e| ChunkError::Fatal(e.to_string()))?;

    // Download entire chunk to memory (like hf_transfer)
    let content = response
        .bytes()
        .await
        .map_err(|e| ChunkError::Retryable(e.to_string()))?;

    // Write to file at correct offset
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(false)
        .create(true)
        .open(filename)
        .await
        .map_err(|e| ChunkError::Fatal(e.to_string()))?;

    file.seek(std::io::SeekFrom::Start(start as u64))
        .await
        .map_err(|e| ChunkError::Fatal(e.to_string()))?;

    file.write_all(&content)
        .await
        .map_err(|e| ChunkError::Fatal(e.to_string()))?;

    Ok(content.len())
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
    chunk_size=10*1024*1024,
    initial_chunk_size=10*1024*1024,
    target_chunk_seconds=0.25,
    min_chunk_size=64*1024,
    max_chunk_size=64*1024*1024,
    callback_min_interval_secs=0.1,
    callback_min_bytes=512*1024
))]
#[allow(clippy::too_many_arguments)]
fn download_from_url(
    py: Python<'_>,
    url: String,
    file_path: String,
    part_path: String,
    headers: Option<&Bound<'_, PyDict>>,
    verify_tls: bool,
    progress_callback: Option<PyObject>,
    #[allow(unused)] adaptive: bool, // kept for API compat
    chunk_size: usize,
    #[allow(unused)] initial_chunk_size: usize,
    #[allow(unused)] target_chunk_seconds: f64,
    #[allow(unused)] min_chunk_size: usize,
    #[allow(unused)] max_chunk_size: usize,
    callback_min_interval_secs: f64,
    callback_min_bytes: usize,
) -> PyResult<String> {
    // Skip if already exists
    if Path::new(&file_path).exists() {
        return Ok(file_path);
    }
    ensure_parent_dir(&file_path)?;

    let mut hm = pydict_to_headermap(headers)?;
    if !hm.contains_key("accept-encoding") {
        hm.insert(
            HeaderName::from_static("accept-encoding"),
            HeaderValue::from_static("identity"),
        );
    }

    // Config from env (hf_transfer style) - use aggressive defaults for throughput
    let max_files = env_usize("APEX_RUST_DOWNLOAD_MAX_FILES", 64).max(1);
    let parallel_failures = env_usize("APEX_RUST_DOWNLOAD_PARALLEL_FAILURES", 3).max(1);
    let max_retries = env_usize("APEX_RUST_DOWNLOAD_MAX_RETRIES", 5);
    // Force minimum 10MB chunks for throughput (matches hf_transfer)
    let chunk_size = env_usize("APEX_RUST_DOWNLOAD_CHUNK_SIZE", chunk_size).max(10 * 1024 * 1024);

    let url2 = url.clone();
    let file_path2 = file_path.clone();
    let part_path2 = part_path.clone();
    let cb = progress_callback;
    let hm2 = hm.clone();
    let name = Path::new(&file_path)
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string());

    py.allow_threads(move || -> PyResult<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        rt.block_on(async move {
            // Build client (like hf_transfer)
            let mut client_builder = reqwest::Client::builder()
                .http2_keep_alive_timeout(Duration::from_secs(15))
                .pool_max_idle_per_host(max_files)
                .redirect(reqwest::redirect::Policy::limited(10));
            if !verify_tls {
                client_builder = client_builder.danger_accept_invalid_certs(true);
            }
            let client = client_builder
                .build()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Separate auth header (like hf_transfer)
            let mut headers = hm2.clone();
            let auth_header = headers.remove(AUTHORIZATION);

            // Initial request to get redirected URL and file size
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
                    let wait = parse_ratelimit_reset(resp.headers())
                        .map(Duration::from_secs)
                        .unwrap_or_else(default_ratelimit_wait);
                    sleep(wait).await;
                    continue;
                }

                break resp
                    .error_for_status()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            };

            let redirected_url = initial.url().to_string();

            // Re-add auth if same host
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
                .ok_or_else(|| PyRuntimeError::new_err("Could not parse file size"))?;

            // Resume bookkeeping
            let ranges_path = format!("{part_path2}.ranges");
            let part_exists = Path::new(&part_path2).exists();
            let ranges_exists = Path::new(&ranges_path).exists();

            let mut completed: Vec<(usize, usize)> = Vec::new();
            if part_exists {
                if !ranges_exists {
                    let _ = fs::remove_file(&part_path2);
                } else {
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

            // Check if already complete
            if !completed.is_empty() {
                if completed.len() == 1 && completed[0].0 == 0 && completed[0].1 + 1 >= length {
                    ensure_parent_dir(&file_path2)?;
                    fs::rename(&part_path2, &file_path2)
                        .map_err(|e| PyOSError::new_err(e.to_string()))?;
                    let _ = fs::remove_file(&ranges_path);
                    if let Some(ref callback) = cb {
                        Python::with_gil(|py| {
                            callback.call1(py, (length as i64, Some(length as i64), name.clone()))?;
                            Ok::<(), PyErr>(())
                        })?;
                    }
                    return Ok::<(), PyErr>(());
                }
            }

            // Pre-create part file
            ensure_parent_dir(&part_path2)?;
            {
                let f = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&part_path2)
                    .await
                    .map_err(|e| PyOSError::new_err(e.to_string()))?;
                f.set_len(length as u64)
                    .await
                    .map_err(|e| PyOSError::new_err(e.to_string()))?;
            }

            // Calculate chunks to download
            let missing = if completed.is_empty() {
                vec![(0usize, length - 1)]
            } else {
                compute_missing_ranges(&completed, length)
            };

            // Split missing ranges into fixed-size chunks
            let mut chunks: Vec<(usize, usize)> = Vec::new();
            for (range_start, range_end) in missing {
                let mut pos = range_start;
                while pos <= range_end {
                    let end = std::cmp::min(pos + chunk_size - 1, range_end);
                    chunks.push((pos, end));
                    pos = end + 1;
                }
            }

            let total_chunks = chunks.len();
            if total_chunks == 0 {
                // Nothing to download
                ensure_parent_dir(&file_path2)?;
                fs::rename(&part_path2, &file_path2)
                    .map_err(|e| PyOSError::new_err(e.to_string()))?;
                let _ = fs::remove_file(&ranges_path);
                return Ok::<(), PyErr>(());
            }

            // Progress tracking
            let done_bytes: u64 = completed
                .iter()
                .map(|(s, e)| (*e as u64).saturating_sub(*s as u64).saturating_add(1))
                .sum();
            let downloaded = Arc::new(AtomicU64::new(done_bytes));
            let total_len = length as u64;
            let cb_interval = Duration::from_secs_f64(callback_min_interval_secs.max(0.0));
            let cb_min_bytes = callback_min_bytes.max(1) as u64;

            // Ranges file for resume - use unbounded channel to avoid blocking
            let (ranges_tx, mut ranges_rx) = tokio::sync::mpsc::unbounded_channel::<(usize, usize)>();
            
            // Background task to batch-write completed ranges
            let ranges_path_clone = ranges_path.clone();
            let ranges_writer = tokio::spawn(async move {
                let mut file = match OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&ranges_path_clone)
                    .await
                {
                    Ok(f) => f,
                    Err(_) => return,
                };
                while let Some((start, stop)) = ranges_rx.recv().await {
                    let _ = file.write_all(format!("{start},{stop}\n").as_bytes()).await;
                }
            });

            // Semaphores (hf_transfer style)
            let semaphore = Arc::new(Semaphore::new(max_files));
            let parallel_failures_semaphore = Arc::new(Semaphore::new(parallel_failures));
            let cancel = Arc::new(AtomicBool::new(false));

            // Spawn ALL chunk tasks upfront (key difference from worker pool)
            let mut handles = FuturesUnordered::new();

            for (start, stop) in chunks {
                let client = client.clone();
                let url = redirected_url.clone();
                let part_path = part_path2.clone();
                let headers = headers.clone();
                let semaphore = semaphore.clone();
                let pf_sem = parallel_failures_semaphore.clone();
                let ranges_tx = ranges_tx.clone();
                let cancel = cancel.clone();

                handles.push(tokio::spawn(async move {
                    // Acquire semaphore (limits concurrent downloads)
                    let permit = semaphore
                        .acquire_owned()
                        .await
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                    let mut attempt = 0usize;
                    let result = loop {
                        if cancel.load(Ordering::Relaxed) {
                            return Err(PyRuntimeError::new_err("Download cancelled"));
                        }

                        match download_chunk(&client, &url, &part_path, start, stop, headers.clone()).await {
                            Ok(wrote) => {
                                // Record completed range (non-blocking send)
                                let _ = ranges_tx.send((start, stop));
                                break Ok(wrote);
                            }
                            Err(ChunkError::RateLimited(wait)) => {
                                sleep(wait).await;
                                // Don't count as retry
                            }
                            Err(ChunkError::Retryable(msg)) => {
                                if attempt >= max_retries {
                                    break Err(PyRuntimeError::new_err(format!(
                                        "Failed after {max_retries} retries: {msg}"
                                    )));
                                }
                                // Acquire failure permit
                                let _pf_permit = pf_sem.clone().try_acquire_owned().map_err(|_| {
                                    PyRuntimeError::new_err(format!(
                                        "Too many parallel failures: {msg}"
                                    ))
                                })?;
                                let wait = exponential_backoff_ms(BASE_WAIT_TIME_MS, attempt, MAX_WAIT_TIME_MS);
                                sleep(Duration::from_millis(wait as u64)).await;
                                attempt += 1;
                            }
                            Err(ChunkError::Fatal(msg)) => {
                                break Err(PyRuntimeError::new_err(msg));
                            }
                        }
                    };

                    drop(permit);
                    result
                }));
            }
            // Drop the main sender so the background writer knows when to stop
            drop(ranges_tx);

            // Process results with progress callback
            let mut last_cb_at = Instant::now();
            let mut last_cb_bytes = downloaded.load(Ordering::Relaxed);

            while let Some(result) = handles.next().await {
                match result {
                    Ok(Ok(size)) => {
                        downloaded.fetch_add(size as u64, Ordering::Relaxed);

                        // Progress callback (throttled)
                        if let Some(ref callback) = cb {
                            let now = Instant::now();
                            let n = downloaded.load(Ordering::Relaxed);
                            if now.duration_since(last_cb_at) >= cb_interval
                                || n.saturating_sub(last_cb_bytes) >= cb_min_bytes
                            {
                                last_cb_at = now;
                                last_cb_bytes = n;
                                Python::with_gil(|py| -> PyResult<()> {
                                    callback.call1(py, (n as i64, Some(total_len as i64), name.clone()))?;
                                    Ok(())
                                })?;
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        cancel.store(true, Ordering::Relaxed);
                        // IMPORTANT: keep partial state on failure so a subsequent attempt can resume.
                        // The Python wrapper will prefer retrying Rust when `{part_path}.ranges` exists.
                        return Err(e);
                    }
                    Err(e) => {
                        cancel.store(true, Ordering::Relaxed);
                        // IMPORTANT: keep partial state on failure so a subsequent attempt can resume.
                        // The Python wrapper will prefer retrying Rust when `{part_path}.ranges` exists.
                        return Err(PyRuntimeError::new_err(format!("Task error: {e}")));
                    }
                }
            }

            // Wait for ranges writer to finish
            let _ = ranges_writer.await;

            // Final progress callback
            downloaded.store(total_len, Ordering::Relaxed);
            if let Some(ref callback) = cb {
                Python::with_gil(|py| {
                    callback.call1(py, (total_len as i64, Some(total_len as i64), name.clone()))?;
                    Ok::<(), PyErr>(())
                })?;
            }

            // Finalize
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
