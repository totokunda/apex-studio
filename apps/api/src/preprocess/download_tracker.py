"""
Download progress tracking for model downloads
"""

from typing import Callable, Optional
from tqdm import tqdm
import time


# Global callback that can be set by tasks
DOWNLOAD_PROGRESS_CALLBACK: Optional[Callable] = None


class DownloadProgressTracker:
    """
    Tracks download progress and reports via websocket
    """

    def __init__(self, job_id: str, progress_callback: Callable):
        self.job_id = job_id
        # Expects callback(progress: float, message: str, metadata: Optional[dict])
        self.progress_callback = progress_callback
        self.files = {}  # Track multiple files
        self.last_update = 0
        self.update_interval = 0.5  # Update every 0.5 seconds

    def update_progress(self, current: int, total: int, filename: str):
        """
        Update progress for a specific file

        Args:
            filename: Name of file being downloaded
            current: Current bytes downloaded
            total: Total bytes to download
        """
        # Throttle updates
        current_time = time.time()
        if not current or not total:
            return
        if current_time - self.last_update < self.update_interval and current < total:
            return

        self.last_update = current_time
        self.files[filename] = {"current": current, "total": total}

        # Calculate overall progress
        total_downloaded = sum((f["current"] or 0) for f in self.files.values())
        total_size = sum((f["total"] or 0) for f in self.files.values())

        if total_size > 0:
            progress = total_downloaded / total_size
            # Map to 0.2-0.9 range (leaving 0-0.2 for initialization, 0.9-1.0 for finalization)
            mapped_progress = 0.2 + (progress * 0.7)

            # Format size
            current_mb = total_downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)

            message = f"Downloading {filename}: {current_mb:.1f}MB / {total_mb:.1f}MB"

            self.progress_callback(
                mapped_progress,
                message,
                {
                    "filename": filename,
                    "current_bytes": current,
                    "total_bytes": total,
                    # compatibility with frontend extractors
                    "downloaded": current,
                    "total": total,
                    "bytes_downloaded": current,
                    "bytes_total": total,
                    "current_mb": round(current_mb, 2),
                    "total_mb": round(total_mb, 2),
                    "files": self.files,
                },
            )


class TqdmProgressHook(tqdm):
    """
    Custom tqdm that reports to download tracker
    """

    def __init__(self, *args, **kwargs):
        # Extract our custom params
        self.filename = kwargs.pop("filename", "unknown")
        self.tracker = kwargs.pop("tracker", None)

        # Initialize tqdm
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        """Override update to report progress"""
        result = super().update(n)

        if self.tracker and self.total:
            self.tracker.update_progress(self.filename, self.n, self.total)

        return result
