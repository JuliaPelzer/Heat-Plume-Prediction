import os
import yaml
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

def beep(case:str="end"):
    duration = 0.05  # seconds
    freq = 440  # Hz
    os.system(f"play -nq -t alsa synth {duration} sine {freq}")
    if case=="end":
        freq = 640  # Hz
        os.system(f"play -nq -t alsa synth {duration} sine {freq}")

def set_paths(dataset_name: str, inputs_prep:str = "", name_extension: str = "", case_2hp: bool = False):
    if os.path.exists("paths.yaml"):
        with open("paths.yaml", "r") as f:
            paths = yaml.load(f, Loader=yaml.SafeLoader)
            default_raw_dir = paths["default_raw_dir"]
            datasets_prepared_dir = paths["datasets_prepared_dir"]
            if case_2hp:
                datasets_prepared_dir = paths["datasets_prepared_dir_2hp"]
            
    elif not os.path.exists("/scratch/sgs/pelzerja/"):
        default_raw_dir = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets/1hp_boxes"
        datasets_prepared_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
    
    dataset_prepared_path = os.path.join(datasets_prepared_dir, dataset_name+"_"+inputs_prep+name_extension)
    if case_2hp:
        dataset_prepared_path = os.path.join(datasets_prepared_dir, dataset_name)
        print(dataset_prepared_path)

    return default_raw_dir, datasets_prepared_dir, dataset_prepared_path

if __name__ == "__main__":
    beep()