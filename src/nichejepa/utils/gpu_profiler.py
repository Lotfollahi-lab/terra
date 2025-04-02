import os
import torch
from torch.profiler import schedule, profile, ProfilerActivity

def create_profiler(save_folder_path: str,
                    wait: int = 2,
                    warmup: int = 2,
                    active: int = 3,
                    repeat: int = 1,
                    use_cuda: bool = True,
                    with_stack: bool = True,
                    with_flops: bool = True,
                    with_modules: bool = True) -> torch.profiler.profile:
    """
    Creates and returns a configured PyTorch profiler instance.

    Args:
        save_folder_path (str): Base directory where profiler trace files will be stored.
        wait (int, optional): Number of steps to wait before profiling starts. Defaults to 2.
        warmup (int, optional): Number of warmup steps. Defaults to 2.
        active (int, optional): Number of active profiling steps. Defaults to 3.
        repeat (int, optional): Number of times to repeat the schedule. Defaults to 1.

    Returns
    -------
        torch.profiler.profile: Configured profiler instance.
    """
    # Create the output directory for trace files if it doesn't exist
    output_dir = os.path.join(save_folder_path, "profiler_traces")
    os.makedirs(output_dir, exist_ok=True)

    # Create a schedule for the profiler: wait, then warmup, then active steps.
    prof_schedule = schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat
    )

    # Initialize and return the profiler with desired configuration
    profiler = profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        with_flops=with_flops,
        with_modules=with_modules,
        use_cuda=use_cuda
    )

    return profiler
