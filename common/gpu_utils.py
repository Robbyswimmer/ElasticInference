import logging
import subprocess

import torch

logger = logging.getLogger(__name__)


def get_gpu_count():
    """Return number of available CUDA GPUs."""
    return torch.cuda.device_count()


def get_gpu_memory(device=0):
    """Return (total_mb, used_mb, free_mb) for a CUDA device."""
    if not torch.cuda.is_available():
        return (0, 0, 0)
    total = torch.cuda.get_device_properties(device).total_mem / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    free = total - reserved
    return (total, allocated, free)


def set_cuda_mps_percentage(percentage):
    """Set CUDA MPS active thread percentage for GPU spatial sharing.

    Must be called before any CUDA context is created.
    Requires nvidia-cuda-mps-control to be running on the host.
    """
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True, check=False,
        )
        subprocess.run(
            f"echo set_active_thread_percentage {percentage} | nvidia-cuda-mps-control",
            shell=True, capture_output=True, check=True,
        )
        logger.info("CUDA MPS active thread percentage set to %d%%", percentage)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning("Could not set MPS percentage: %s", e)


def log_gpu_info():
    """Log available GPU information."""
    if not torch.cuda.is_available():
        logger.info("No CUDA GPUs available, using CPU")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total, used, free = get_gpu_memory(i)
        logger.info(
            "GPU %d: %s — %.0fMB total, %.0fMB used, %.0fMB free",
            i, props.name, total, used, free,
        )
