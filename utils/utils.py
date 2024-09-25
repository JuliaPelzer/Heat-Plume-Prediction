import os
#from pynvml import *
import re
from typing import List

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

def beep(case:str="end"):
    duration = 0.05  # seconds
    freq = 440  # Hz
    os.system(f"play -nq -t alsa synth {duration} sine {freq}")
    if case=="end":
        freq = 640  # Hz
        os.system(f"play -nq -t alsa synth {duration} sine {freq}")

def re_split_number_text(input: str) -> List:
    match = re.match(r"([a-z]+)([0-9]+)", input, re.I)
    if match:
        items = match.groups()
    return items