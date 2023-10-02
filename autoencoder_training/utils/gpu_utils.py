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

        free_memory = gpu_info.free / one_gb

        if gpu_info.free > max_memory:
            max_memory = gpu_info.free
            max_gpu_index = i

    pynvml.nvmlShutdown()

    if free_memory > min_memory:
        return max_gpu_index
    else:
        print("No NVIDIA GPUs with sufficient memory found.")
        return None