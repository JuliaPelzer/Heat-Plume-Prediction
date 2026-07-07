import argparse
from asyncio import run
from shutil import copytree
import time
import signal
from pathlib import Path
import torch
from tqdm.auto import tqdm
import numpy as np

from streamlines_helpers import make_streamlines, save_new_datapoint, correct_info, extend_inputs_dims
from utils.utils_args import load_yaml
from preprocessing.transforms import NormalizeTransform

def build_streamlines(dataset_path:Path=None, **kwargs):
    ## copy ixydk files (later overwrite xyd)
    destination = dataset_path.parent/f"{dataset_path.name}+s(t0,Q,Ttrend)"
    copytree(dataset_path,destination, dirs_exist_ok=True)

    prop_is = {"vx": 3,
            "vy" : 4,
            "Q" : 5,
            "sf" : 6,
            "sf_outers" : 7,
            "s_seasonal" : 8
            }

    norm_before = NormalizeTransform(load_yaml(destination/"info.yaml"))
    correct_info(destination, i_s=prop_is["sf"], i_s_outer=prop_is["sf_outers"], i_s_seasonal=prop_is["s_seasonal"])
    norm_after = NormalizeTransform(load_yaml(destination/"info.yaml"))

    streamline_timeout = 60
    def _timeout_handler(signum, frame):
        raise TimeoutError("streamline calculation timed out")
    
    # Square root regression (fitted on step1-train data), but if result would be too small, set width to 2 cells
    width_model = lambda Q: max(0.84 * Q**0.5 -1.32, 2)
    temperature_injection_series = np.load(dataset_path/"temperature_injection_series.npy")

    runs_tqdm = tqdm([run for run in (destination / "Inputs").iterdir() if run.name.endswith(".pt")], desc="Processing runs")
    for run in runs_tqdm:
        print(run.stem)
        runs_tqdm.set_postfix_str(f"{run.stem}")
        start_time = time.time()
        inputs = torch.load(run)
        inputs = norm_before.reverse(inputs, "Inputs")
        inputs = extend_inputs_dims(inputs, n_added_inputs=3) # add 3 channels for streamlines
        inputs_reduced = inputs.numpy()

        # make streamlines
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(streamline_timeout)
        try:
            streamlines_faded = make_streamlines(Qinj2D=inputs_reduced[prop_is["Q"]], vx=inputs_reduced[prop_is["vx"]], vy=inputs_reduced[prop_is["vy"]], dims=inputs_reduced[0].shape, faded=True, **kwargs)

            streamlines_faded_top = make_streamlines(Qinj2D=inputs_reduced[prop_is["Q"]], vx=inputs_reduced[prop_is["vx"]], vy=inputs_reduced[prop_is["vy"]], dims=inputs_reduced[0].shape, faded=False, offset={"model": width_model, "factor": 1}, **kwargs)
            streamlines_faded_bottom = make_streamlines(Qinj2D=inputs_reduced[prop_is["Q"]], vx=inputs_reduced[prop_is["vx"]], vy=inputs_reduced[prop_is["vy"]], dims=inputs_reduced[0].shape, faded=False, offset={"model": width_model, "factor": -1}, **kwargs)

            streamlines_seasonal = make_streamlines(Qinj2D=inputs_reduced[prop_is["Q"]], vx=inputs_reduced[prop_is["vx"]], vy=inputs_reduced[prop_is["vy"]], dims=inputs_reduced[0].shape, faded=False, seasonal=True, temperature_series=temperature_injection_series, **kwargs)
        except TimeoutError:
            print(f"Skipping {run.stem} after {time.time()-start_time} seconds: streamline calculation timed out")
            continue
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # norm inputs acc. to info
        inputs_normed = norm_after(torch.tensor(inputs_reduced), "Inputs")
        inputs_normed[prop_is["sf"]] = streamlines_faded.unsqueeze(0)
        inputs_normed[prop_is["sf_outers"]] = torch.max(streamlines_faded_top.unsqueeze(0), streamlines_faded_bottom.unsqueeze(0))
        inputs_normed[prop_is["s_seasonal"]] = streamlines_seasonal.unsqueeze(0)

        save_new_datapoint(destination, run, inputs_normed)


if __name__ == "__main__":
    PATH_DATA_PREP = Path("/scratch/sgs/pelzerja/datasets_prepared/bm/")

    # argparse for dataset_name with default
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="step1_overfit1", help="Name of the dataset folder in PATH_DATA_PREP.")
    args = parser.parse_args()
    dataset_name = args.data

    # STEP 2: calculate streamlines with simulated or with predicted velocity fields
    build_streamlines(PATH_DATA_PREP / dataset_name, method="Radau")

