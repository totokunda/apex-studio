import os
import subprocess
import ray
from loguru import logger


def _gpu_mem_info_torch():
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        infos = []
        for i in range(torch.cuda.device_count()):
            # Make device current, then ask PyTorch for mem info
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()  # bytes
            infos.append({"index": i, "free": free, "total": total})
        return infos
    except Exception:
        return None


def _gpu_mem_info_nvml():
    try:
        import pynvml as nvml

        nvml.nvmlInit()
        n = nvml.nvmlDeviceGetCount()
        infos = []
        for i in range(n):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            infos.append({"index": i, "free": mem.free, "total": mem.total})
        nvml.nvmlShutdown()
        return infos
    except Exception:
        return None


def _gpu_mem_info_nvidia_smi():
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .splitlines()
        )
        infos = []
        for i, line in enumerate(out):
            total_mb, used_mb = [int(x.strip()) for x in line.split(",")]
            free_mb = max(total_mb - used_mb, 0)
            # convert MB to bytes for consistency
            infos.append(
                {"index": i, "free": free_mb * 1024**2, "total": total_mb * 1024**2}
            )
        return infos if infos else None
    except Exception:
        return None


def _on_mps():
    try:
        import torch

        # Treat MPS as a single logical accelerator
        return (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def get_best_gpu():
    """
    Determine the best GPU to use for a task.

    Priority:
      1) If Apple MPS is available -> device 0
      2) If CUDA GPUs exist -> pick GPU with MOST FREE VRAM
      3) Else -> None (use CPU)

    Returns:
        Tuple of (device_index: int or None, device_type: str)
    """
    # 1) MPS (Apple Silicon)
    if _on_mps():
        logger.info("Using MPS device (Apple Silicon)")
        return 0, "mps"

    # 2) CUDA mem info via PyTorch, then NVML, then nvidia-smi
    infos = _gpu_mem_info_torch() or _gpu_mem_info_nvml() or _gpu_mem_info_nvidia_smi()

    if infos:
        # choose the device with maximum free memory
        best = max(infos, key=lambda d: d["free"])
        logger.info(
            f"Using CUDA device {best['index']} with {best['free'] / (1024**3):.2f} GB free"
        )
        return best["index"], "cuda"

    # 3) Fallback to CPU
    logger.info("No GPU available, using CPU")
    return None, "cpu"


def get_ray_resources(
    device_index: int = None,
    device_type: str = "cuda",
    load_profile: str = "light",
):
    """
    Get Ray resource requirements dynamically based on current availability.

    Behavior:
      - CPU: allocate CPUs based on load profile.
      - MPS: treat as CPU-only for Ray scheduling.
      - CUDA: **always request a full GPU (num_gpus=1.0)** when a CUDA device
        is selected. This ensures Ray queues tasks when the GPU is busy, instead
        of falling back to CPU or oversubscribing via fractional GPUs.
        If the cluster exposes per-GPU custom resources like `GPU_{index}`, we
        honor them to best-effort pin placement when `device_index` is provided.

    Args:
        device_index: GPU index to use (None means CPU when device_type is "cpu").
        device_type: "cuda", "mps", or "cpu".
        load_profile: CPU load profile: "light" (≈1 CPU), "medium" (more CPUs),
            or "heavy" (most of the machine's CPUs).

    Returns:
        Dictionary of Ray resource requirements for task/actor scheduling.
    """
    # Normalize CPU load profile
    profile = (load_profile or "light").lower()
    if profile not in {"light", "medium", "heavy", "all"}:
        profile = "light"

    # Ray may not be initialized in all code paths; guard calls accordingly.
    try:
        ray_initialized = ray.is_initialized()
    except Exception:
        ray_initialized = False

    def _select_cpus(cpu_avail: float, total_cpus: float, profile: str) -> float:
        """
        Choose how many CPUs to request based on availability and profile.

        light  -> ~1 CPU (or a small fraction if less is left)
        medium -> a few CPUs (2–4) when available
        heavy  -> most of the remaining CPUs (leave a little headroom)
        """
        if total_cpus is None or total_cpus <= 0:
            total_cpus = 1.0
        if cpu_avail <= 0:
            # Preserve previous behavior: default to 1 CPU when Ray reports none,
            # but still respect the load profile relative to total_cpus.
            cpu_avail = float(total_cpus)

        # Medium: use a moderate fraction of the machine's CPUs
        if profile == "medium":
            target = max(2.0, total_cpus * 0.3)  # at least 2 CPUs or 25% of machine
            return min(cpu_avail, target)

        # Heavy: use most of the machine's CPUs, leaving a little headroom
        if profile == "heavy":
            target = max(1.0, total_cpus * 0.9)  # ~90% of machine
            return min(cpu_avail, target)

        if profile == "all":
            return cpu_avail

        # Default: light profile -> ~1 CPU
        return 1.0 if cpu_avail >= 1.0 else max(0.25, cpu_avail)

    # Helper to read available/cluster resources if Ray is up
    available = {}
    cluster = {}
    if ray_initialized:
        try:
            available = ray.available_resources() or {}
            cluster = ray.cluster_resources() or {}
        except Exception:
            available, cluster = {}, {}

    # CPU-only path or explicit CPU request
    if device_type == "cpu" or device_index is None:
        if ray_initialized:
            cpu_avail = float(available.get("CPU", 0.0))
            total_cpus = float(cluster.get("CPU", cpu_avail)) or cpu_avail
            num_cpus = _select_cpus(cpu_avail, total_cpus, profile)
            return {"num_cpus": num_cpus}
        # Fallback when Ray isn't initialized
        total_cpus = os.cpu_count() or 1
        num_cpus = _select_cpus(float(total_cpus), float(total_cpus), profile)
        return {"num_cpus": num_cpus}

    # Apple MPS: Ray does not track an MPS resource; treat as CPU-only scheduling
    if device_type == "mps":
        if ray_initialized:
            cpu_avail = float(available.get("CPU", 0.0))
            total_cpus = float(cluster.get("CPU", cpu_avail)) or cpu_avail
            num_cpus = _select_cpus(cpu_avail, total_cpus, profile)
            return {"num_cpus": num_cpus, "num_gpus": 0}
        total_cpus = os.cpu_count() or 1
        num_cpus = _select_cpus(float(total_cpus), float(total_cpus), profile)
        return {"num_cpus": num_cpus, "num_gpus": 0}

    # CUDA path
    if device_type == "cuda":
        if ray_initialized:
            cpu_avail = float(available.get("CPU", 0.0))
            gpu_avail = float(available.get("GPU", 0.0))

            # CPU request: shape based on load profile and machine size
            total_cpus = float(cluster.get("CPU", cpu_avail)) or cpu_avail
            num_cpus = _select_cpus(cpu_avail, total_cpus, profile)

            # GPU request: always request a full GPU so Ray queues when the GPU is busy.
            # Only fall back to CPU-only if the cluster has *no* GPU capacity at all.
            total_gpus = float(cluster.get("GPU", gpu_avail) or 0.0)
            if total_gpus <= 0.0:
                logger.warning(
                    "CUDA selected but Ray cluster reports no GPU capacity; "
                    "falling back to CPU-only scheduling."
                )
                return {"num_cpus": num_cpus}
            num_gpus = 1.0

            resources = None
            # If the cluster exposes per-GPU custom resources, honor them to pin the task
            if device_index is not None:
                custom_key = f"GPU_{device_index}"
                if custom_key in cluster:
                    # Reserve the custom resource fully to ensure placement on the target GPU
                    resources = {custom_key: 1}

            result = {"num_cpus": num_cpus, "num_gpus": num_gpus}
            if resources:
                result["resources"] = resources
            return result

        # Fallback when Ray isn't initialized: assume single-node machine
        total_cpus = os.cpu_count() or 1
        num_cpus = _select_cpus(float(total_cpus), float(total_cpus), profile)
        return {"num_cpus": num_cpus, "num_gpus": 1}

    # Unknown device type -> default to 1 CPU
    return {"num_cpus": 1}
