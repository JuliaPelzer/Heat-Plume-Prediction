from pynvml import *

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def get_memory_usage(device):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(device)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {sizeof_fmt(info.total)}')
    print(f'free     : {sizeof_fmt(info.free)}')
    print(f'used     : {sizeof_fmt(info.used)}')