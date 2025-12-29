# PyTorch Memory Management System

A comprehensive memory management module for PyTorch that provides intelligent offloading between GPU, CPU, and disk storage based on memory usage patterns.

## Features

- **Automatic Memory Monitoring**: Real-time tracking of GPU, CPU, and disk memory usage
- **Intelligent Offloading**: Smart decisions on when and where to offload model parameters
- **Multiple Storage Tiers**: Support for GPU → CPU → Disk offloading hierarchy
- **Module Wrapping**: Transparent wrapping of PyTorch modules with minimal code changes
- **Eager Offload on Wrap**: Automatically offload wrapped modules so `.to()` calls never spike VRAM
- **Smart Bias Handling**: Keep 1D parameters (biases, LayerNorm scales) resident by default for stability
- **Configurable Thresholds**: Customizable memory thresholds and behavior
- **Performance Optimized**: Designed for minimal latency impact on model inference
- **Thread-Safe**: Safe for multi-threaded applications

## Quick Start

```python
import torch
import torch.nn as nn
from memory_management import auto_manage_model, MemoryConfig

# Create your model
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.Linear(1024, 10)
)

# Automatically wrap with memory management
manager = auto_manage_model(model)

# Use your model normally - memory management happens automatically
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Wrapped layers are offloaded immediately by default, so calling `.to(device)`
# is safe even if the model checkpoint is larger than available VRAM.

x = torch.randn(32, 1024, device=device)
output = model(x)

# Cleanup when done
manager.cleanup()
```

## Installation

Install the required dependencies:

```bash
pip install torch>=1.9.0 psutil>=5.8.0
```

## Configuration

The system can be configured for different use cases:

### Low Memory Environment
```python
config = MemoryConfig.for_low_memory()
manager = auto_manage_model(model, config)
```

### High Performance Environment
```python
config = MemoryConfig.for_high_performance()
manager = auto_manage_model(model, config)
```

### Custom Configuration
```python
config = MemoryConfig(
    gpu_offload_threshold=0.85,  # Start offloading at 85% GPU usage
    cpu_offload_threshold=0.90,  # Offload to disk at 90% CPU usage
    memory_check_interval=0.1,   # Check every 100ms
    enable_disk_offload=True,    # Enable disk storage
    max_disk_usage_gb=50.0       # Max 50GB disk cache
)
manager = MemoryManager(config)
```

## Advanced Usage

### Manual Module Wrapping
```python
from memory_management import MemoryManager, MemoryConfig

config = MemoryConfig()
manager = MemoryManager(config)
manager.start()

# Wrap specific modules
linear_layer = nn.Linear(1024, 512)
wrapped_layer = manager.wrap_module(linear_layer, module_id="my_layer")

# Manual control
wrapped_layer.offload_all()  # Force offload
wrapped_layer.load_all()     # Force reload
```

### Layer Type Selection
```python
# Only wrap Linear layers (default)
manager.wrap_model(model, layer_types=[nn.Linear])

# Wrap multiple layer types
manager.wrap_model(model, layer_types=[nn.Linear, nn.Conv2d, nn.LSTM])
```

### Memory Monitoring
```python
# Get detailed memory summary
summary = manager.get_memory_summary()
print(f"GPU usage: {summary['gpu']['usage_ratio']:.2%}")
print(f"Managed modules: {summary['system']['managed_modules']}")
print(f"Total offloads: {summary['system']['total_offloads']}")
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_offload_threshold` | 0.85 | GPU usage ratio to start offloading |
| `gpu_emergency_threshold` | 0.95 | GPU usage ratio for emergency offloading |
| `gpu_reload_threshold` | 0.70 | GPU usage ratio to allow reloading |
| `cpu_offload_threshold` | 0.90 | CPU usage ratio to offload to disk |
| `memory_check_interval` | 0.1 | Memory check frequency (seconds) |
| `enable_cpu_offload` | True | Enable CPU offloading |
| `enable_disk_offload` | True | Enable disk offloading |
| `compress_disk_cache` | True | Compress tensors on disk |
| `eager_offload_on_wrap` | True | Immediately offload wrapped modules so `.to()` doesn't blow up VRAM |
| `offload_1d_parameters` | False | Whether to offload 1D tensors like bias vectors |
| `max_disk_usage_gb` | 50.0 | Maximum disk cache size |
| `offload_batch_size` | 10 | Max modules to offload per batch |
| `prefetch_enabled` | True | Enable predictive loading |

## Performance Considerations

### For Low Latency Applications
- Set higher GPU thresholds to keep more data in GPU memory
- Reduce `memory_check_interval` for faster response
- Enable `prefetch_enabled` for predictive loading
- Use smaller `offload_batch_size` for granular control

### For Memory-Constrained Environments
- Set lower GPU and CPU thresholds
- Enable disk offloading with compression
- Increase `offload_batch_size` for efficiency
- Consider disabling less critical features

## Examples

See `examples.py` for comprehensive usage examples including:
- Basic usage with automatic management
- Custom configuration for different scenarios
- Manual control and fine-tuning
- Transformer model integration
- Sentiment analysis optimization (< 100ms latency)

## Architecture

The system consists of several key components:

1. **MemoryMonitor**: Tracks GPU, CPU, and disk usage in real-time
2. **OffloadStrategies**: Handle moving tensors between storage tiers
3. **ModuleWrapper**: Transparently wraps PyTorch modules
4. **MemoryManager**: Coordinates all components and provides the main interface

## Thread Safety

The system is designed to be thread-safe and can be used in multi-threaded environments. All critical sections are properly synchronized.

## Limitations

- Currently optimized for NVIDIA GPUs with CUDA
- Disk offloading adds latency for offloaded parameters
- Some overhead from memory monitoring (typically < 1% performance impact)
- Requires careful tuning for optimal performance in specific use cases

## Contributing

The system is designed to be modular and extensible. Key areas for enhancement:
- Additional offloading strategies (cloud storage, compression algorithms)
- Better prediction algorithms for proactive offloading
- Integration with model parallelism frameworks
- Support for additional hardware accelerators

## License

This memory management system is provided as-is for research and development purposes. 