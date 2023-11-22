import pynvml

def find_gpu_with_most_memory(min_memory: int = 10):
    pynvml.nvmlInit()

    one_gb = 1024*1024*1024
    device_count = pynvml.nvmlDeviceGetCount()

    if device_count == 0:
        print("No NVIDIA GPUs found.")
        return None

    max_memory = 0
    max_gpu_index = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        free_memory = round(gpu_info.free / one_gb, 2)

        if free_memory > max_memory:
            max_memory = free_memory
            max_gpu_index = i

    pynvml.nvmlShutdown()

    if max_memory > min_memory:
        print(f'Found GPU {max_gpu_index} with {max_memory} GB available.')
        return f'cuda:{max_gpu_index}'
    else:
        print("No NVIDIA GPUs with sufficient memory found.")
        return None