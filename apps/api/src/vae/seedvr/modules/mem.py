import torch

import psutil
import sys
import time
from typing import Dict, Any, Optional

from loguru import logger

global _os_memory_lib
_os_memory_lib = None
import gc

def get_basic_vram_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get basic VRAM availability info (free and total memory).
    Used for capacity planning and initial checks.
    
    Args:
        device: Optional device to query. If None, uses cuda:0
    
    Returns:
        dict: {"free_gb": float, "total_gb": float} or {"error": str}
    """
    try:
        if torch.cuda.is_available():
            if device is None:
                device = torch.device("cuda:0")
            elif not isinstance(device, torch.device):
                device = torch.device(device)
            free_memory, total_memory = torch.cuda.mem_get_info(device)
        elif torch.backends.mps.is_available():
            # MPS doesn't support per-device queries or mem_get_info
            # Use system memory as proxy
            mem = psutil.virtual_memory()
            free_memory = mem.total - mem.used
            total_memory = mem.total
        else:
            return {"error": "No GPU backend available (CUDA/MPS)"}
        
        return {
            "free_gb": free_memory / (1024**3),
            "total_gb": total_memory / (1024**3)
        }
    except Exception as e:
        return {"error": f"Failed to get memory info: {str(e)}"}


def clear_memory(debug: Optional['Debug'] = None, deep: bool = False, force: bool = True, 
                timer_name: Optional[str] = None) -> None:
    """
    Clear memory caches with two-tier approach for optimal performance.
    
    Args:
        debug: Debug instance for logging (optional)
        force: If True, always clear. If False, only clear when <5% free
        deep: If True, perform deep cleanup including GC and OS operations.
              If False (default), only perform minimal GPU cache clearing.
        timer_name: Optional suffix for timer names to make them unique per invocation
    
    Two-tier approach:
        - Minimal mode (deep=False): GPU cache operations (~1-5ms)
          Used for frequent calls during batch processing
        - Deep mode (deep=True): Complete cleanup with GC and OS operations (~10-50ms)
          Used at key points like model switches or final cleanup
    """
    global _os_memory_lib
    
    # Create unique timer names if suffix provided
    if timer_name:
        main_timer = f"memory_clear_{timer_name}"
        gpu_timer = f"gpu_cache_clear_{timer_name}"
        gc_timer = f"garbage_collection_{timer_name}"
        os_timer = f"os_memory_release_{timer_name}"
        completion_msg = f"clear_memory() completion ({timer_name})"
    else:
        main_timer = "memory_clear"
        gpu_timer = "gpu_cache_clear"
        gc_timer = "garbage_collection"
        os_timer = "os_memory_release"
        completion_msg = "clear_memory() completion"
    
    # Start timer for entire operation
    if debug:
        debug.start_timer(main_timer)

    # Check if we should clear based on memory pressure
    if not force:
        should_clear = False
        
        # Use existing function for memory info
        mem_info = get_basic_vram_info(device=None)
        
        if "error" not in mem_info and mem_info["total_gb"] > 0:
            # Check VRAM/MPS memory pressure (5% free threshold)
            free_ratio = mem_info["free_gb"] / mem_info["total_gb"]
            if free_ratio < 0.05:
                should_clear = True
                if debug:
                    backend = "Unified Memory" if torch.backends.mps.is_available() else "VRAM"
                    debug.log(f"{backend} pressure: {mem_info['free_gb']:.2f}GB free of {mem_info['total_gb']:.2f}GB", category="memory")
        
        # For non-MPS systems, also check system RAM separately
        if not should_clear and not torch.backends.mps.is_available():
            mem = psutil.virtual_memory()
            if mem.available < mem.total * 0.05:
                should_clear = True
                if debug:
                    debug.log(f"RAM pressure: {mem.available/(1024**3):.2f}GB free of {mem.total/(1024**3):.2f}GB", category="memory")
        
        if not should_clear:
            # End timer before early return to keep stack clean
            if debug:
                debug.end_timer(main_timer)
            return
    
    # Determine cleanup level
    cleanup_mode = "deep" if deep else "minimal"
    if debug:
        debug.log(f"Clearing memory caches ({cleanup_mode})...", category="cleanup")
    
    # ===== MINIMAL OPERATIONS (Always performed) =====
    # Step 1: Clear GPU caches - Fast operations (~1-5ms)
    if debug:
        debug.start_timer(gpu_timer)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    if debug:
        debug.end_timer(gpu_timer, "GPU cache clearing")

    # ===== DEEP OPERATIONS (Only when deep=True) =====
    if deep:
        # Step 2: Deep garbage collection (expensive ~5-20ms)
        if debug:
            debug.start_timer(gc_timer)

        gc.collect(2)

        if debug:
            debug.end_timer(gc_timer, "Garbage collection")

        # Step 3: Return memory to OS (platform-specific, ~5-30ms)
        if debug:
            debug.start_timer(os_timer)

        try:
            if sys.platform == 'linux':
                # Linux: malloc_trim
                import ctypes  # Import only when needed
                if _os_memory_lib is None:
                    _os_memory_lib = ctypes.CDLL("libc.so.6")
                _os_memory_lib.malloc_trim(0)
                
            elif sys.platform == 'win32':
                # Windows: Trim working set
                import ctypes  # Import only when needed
                if _os_memory_lib is None:
                    _os_memory_lib = ctypes.windll.kernel32
                handle = _os_memory_lib.GetCurrentProcess()
                _os_memory_lib.SetProcessWorkingSetSize(handle, -1, -1)
                
            elif torch.backends.mps.is_available():
                # macOS with MPS
                import ctypes  # Import only when needed
                import ctypes.util
                if _os_memory_lib is None:
                    libc_path = ctypes.util.find_library('c')
                    if libc_path:
                        _os_memory_lib = ctypes.CDLL(libc_path)
                
                if _os_memory_lib:
                    _os_memory_lib.sync()
        except Exception as e:
            if debug:
                debug.log(f"Failed to perform OS memory operations: {e}", level="WARNING", category="memory", force=True)

        if debug:
            debug.end_timer(os_timer, "OS memory release")
    
    # End overall timer
    if debug:
        debug.end_timer(main_timer, completion_msg)

def retry_on_oom(func, *args, debug=None, operation_name="operation", **kwargs):
    """
    Execute function with single OOM retry after memory cleanup.
    
    Args:
        func: Callable to execute
        *args: Positional arguments for func
        debug: Debug instance for logging (optional)
        operation_name: Name for logging
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func(*args, **kwargs)
    """
    try:
        return func(*args, **kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Only handle OOM errors
        if not any(x in str(e).lower() for x in ["out of memory", "allocation on device"]):
            raise
        
        if debug:
            debug.log(f"OOM during {operation_name}: {e}", level="WARNING", category="memory", force=True)
            debug.log(f"Clearing memory and retrying", category="info", force=True)
        
        # Clear memory
        clear_memory(debug=debug, deep=True, force=True, timer_name=operation_name)
        # Let memory settle
        time.sleep(0.5)
        if debug:
            debug.log_memory_state("After memory clearing", show_tensors=False, detailed_tensors=False)
        
        # Single retry
        try:
            result = func(*args, **kwargs)
            if debug:
                debug.log(f"Retry successful for {operation_name}", category="success", force=True)
            return result
        except Exception as retry_e:
            if debug:
                debug.log(f"Retry failed for {operation_name}: {retry_e}", level="ERROR", category="memory", force=True)
            raise