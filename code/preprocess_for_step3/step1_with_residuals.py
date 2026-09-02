"""
Apply the trained residual model (BEST_residuals_exclude11_13_wf_MSE_allData) to the dataset
step3_predV_residual_exclude11_13_wf, unnormalize the predicted velocity residuals, add them to the
unnormalized background velocities (Inputs indices 3 & 4) and store everything as a new dataset
step3_prededV_based_on_residual_exclude11_13_wf with combined new min/max normalization per channel.
Labels (+ their normalization info) are taken from step3(t0). Afterwards, streamlines are built on
the new dataset via step2_streamlines.streamlines_main.build_streamlines.

Run: python step3_prededV_from_residuals.py [--test]
Re-normalize an existing dataset to a provided info.yaml and store it next to it:
    python step3_prededV_from_residuals.py --renorm-data <dataset_path> --new-info <path/to/new_info.yaml> [--destination <path>]
"""

import argparse
import sys
from pathlib import Path
import shutil

CODE_DIR = Path("/home/pelzerja/pelzerja/test_nn/LGCNN_bm/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch
from tqdm.auto import tqdm

from processing.networks.unetVariants import UNet
from step2_streamlines.streamlines_main import build_streamlines
from utils.utils_args import load_yaml, save_yaml

def stats_by_index(section: dict) -> dict:
    return {entry["index"]: entry for entry in section.values()}


def names_by_index(section: dict) -> dict:
    return {entry["index"]: name for name, entry in section.items()}


def unnorm(channel: torch.Tensor, stats: dict) -> torch.Tensor:
    norm_type = stats.get("norm", "Rescale")
    if norm_type == "Rescale":
        return channel * (stats["max"] - stats["min"]) + stats["min"]
    raise ValueError(f"Unsupported normalization type '{norm_type}'")


def predict_and_assemble(pred_residuals:bool, data_dir:Path, model_dir:Path, i_vx:int, i_vy:int) -> tuple[dict, dict]:
    """Apply the model, unnorm residuals and background velocities, assemble new 6-channel inputs."""
    info = load_yaml(data_dir / "info.yaml")
    hps = load_yaml(model_dir / "command_line_arguments.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=len(info["Inputs"]), out_channels=2, #len(info["Labels"]),
                 depth=hps["depth"], init_features=hps["init_features"], kernel_size=hps["kernel_size"],
                 stride=hps["stride"], dilation=hps["dilation"], activation=hps["activation_fct"],
                 norm=hps["norm"], repeat_inner=hps["repeat_inner"], last_activation=hps["last_activation"]).float()
    model.load(model_dir, device)

    in_stats = stats_by_index(info["Inputs"])
    label_stats = stats_by_index(info["Labels"])
    run_names = sorted(f.name for f in (data_dir / "Inputs").iterdir() if f.suffix == ".pt")
    print(run_names)
    new_inputs = {}
    for name in tqdm(run_names, desc="Predicting velocity residuals"):
        x = torch.load(data_dir / "Inputs" / name, weights_only=False)
        try:
            residuals = model.infer(x.unsqueeze(0), device).squeeze(0).cpu()
        except torch.cuda.OutOfMemoryError:
            if device != "cpu":
                print("CUDA out of memory, falling back to CPU for inference.")
                device = "cpu"
                model.to("cpu")
                torch.cuda.empty_cache()
                residuals = model.infer(x.unsqueeze(0), device).squeeze(0).cpu()
            else:
                raise

        # unnorm residuals (normalized like the labels) and add to unnormed background velocities
        if pred_residuals:
            pred_vx = unnorm(x[i_vx], in_stats[i_vx]) + unnorm(residuals[0], label_stats[0])
            pred_vy = unnorm(x[i_vy], in_stats[i_vy]) + unnorm(residuals[1], label_stats[1])
        else:
            pred_vx = unnorm(residuals[0], label_stats[0])
            pred_vy = unnorm(residuals[1], label_stats[1])

        # assemble fully physical inputs (rest as before, no indices above 5)
        new_inputs[name] = torch.stack([unnorm(x[0], in_stats[0]), unnorm(x[1], in_stats[1]),
                                        unnorm(x[2], in_stats[2]), pred_vx, pred_vy,
                                        unnorm(x[5], in_stats[5])])
    return new_inputs, info


def combined_min_max(new_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    run_names = sorted(new_inputs.keys())
    mins = torch.stack([new_inputs[name].amin(dim=(1, 2)) for name in run_names]).amin(dim=0)
    maxs = torch.stack([new_inputs[name].amax(dim=(1, 2)) for name in run_names]).amax(dim=0)
    assert bool((maxs > mins).all()), "min/max degenerate for at least one channel"
    return mins, maxs


def normalize(inputs_raw: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    return (inputs_raw - mins[:, None, None]) / (maxs - mins)[:, None, None]


def build_new_dataset(new_inputs: dict, info_old: dict, destination: Path, mins: torch.Tensor, maxs: torch.Tensor, model_dir:Path, t0_dir:Path, i_vx:int, i_vy:int):
    """Normalize with combined new min/max per channel, store inputs, labels and info.yaml."""
    run_names = sorted(new_inputs.keys())

    old_names = names_by_index(info_old["Inputs"])
    new_names = {0: old_names[0], 1: old_names[1], 2: old_names[2],
                 i_vx: f"{old_names[i_vx]} - predicted by '{model_dir.name}'",
                 i_vy: f"{old_names[i_vy]} - predicted by '{model_dir.name}'",
                 5: old_names[5]}
    new_info = {
        "CellsSize": info_old["CellsSize"],
        "Inputs": {new_names[i]: {"index": i, "min": float(mins[i]), "max": float(maxs[i])} for i in range(6)},
        "Labels": load_yaml(t0_dir / "info.yaml")["Labels"],
    }

    (destination / "Inputs").mkdir(parents=True)
    (destination / "Labels").mkdir(parents=True)
    for name in tqdm(run_names, desc="Saving normalized inputs"):
        torch.save(normalize(new_inputs[name], mins, maxs), destination / "Inputs" / name)

    for name in run_names:  # already normalized in step3(t0), stays that way
        shutil.copy(t0_dir / "Labels" / name, destination / "Labels" / name)
    shutil.copy(t0_dir / "temperature_injection_series.npy", destination / "temperature_injection_series.npy")  # needed by build_streamlines
    save_yaml(new_info, destination / "info.yaml")
    print(f"Stored new dataset under {destination}")

def step1_with_residuals(pred_residuals:bool, step3b_case: bool, i_vx: int, i_vy: int, MODEL_DIR: Path, DATA_DIR: Path, T0_DIR: Path, destination: Path):

    new_inputs, info_old = predict_and_assemble(pred_residuals, DATA_DIR, MODEL_DIR, i_vx, i_vy)
    if step3b_case:
        new_inputs = {sorted(new_inputs.keys())[0]: new_inputs[sorted(new_inputs.keys())[0]]}

    mins, maxs = combined_min_max(new_inputs)
    build_new_dataset(new_inputs, info_old, destination, mins, maxs, MODEL_DIR, T0_DIR, i_vx, i_vy)

    # sanity checks: saved data matches the assembled physical values, and the unchanged
    # channels reproduce the source normalization (stored stats == actual data extremes)
    name = sorted(new_inputs.keys())[0]
    expected = normalize(new_inputs[name], mins, maxs)
    torch.testing.assert_close(torch.load(destination / "Inputs" / name, weights_only=False), expected,
                            msg=lambda msg: f"Saved inputs deviate from assembled inputs:\n{msg}")
    del new_inputs

def main():
    PATH_DATA_PREP = Path("../datasets_prepared/bm")

    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", action="store_true", help="whether residuals were predicted or the full velocity field directly")
    parser.add_argument("--test", action="store_true", help="only process one datapoint and skip streamlines")
    parser.add_argument("--model-dir", type=Path, default=Path("../runs/s3v-BEST_residuals_exclude11_13_wf_MSE_allData"), help="Directory containing the trained model",)
    parser.add_argument("--dataset-name", type=Path, help="Input data directory, relative to PATH_DATA_PREP",)
    parser.add_argument("--t0-dir", type=Path, default="step3_TEST", help="Initial/background data directory, relative to PATH_DATA_PREP",)
    parser.add_argument("--new-dir", type=Path, help="Output data directory, relative to PATH_DATA_PREP",)
    parser.add_argument("--i-vx", type=int, default=3, help="Channel index of the background x-velocity",)
    parser.add_argument("--i-vy", type=int, default=4, help="Channel index of the background y-velocity",)
    args = parser.parse_args()

    DATA_DIR = PATH_DATA_PREP / args.dataset_name 
    T0_DIR = PATH_DATA_PREP /  args.t0_dir 
    NEW_DIR = PATH_DATA_PREP / args.new_dir 
    destination = Path(f"{NEW_DIR}_test") if args.test else NEW_DIR

    step1_with_residuals(args.residuals, args.test, args.i_vx, args.i_vy, Path(args.model_dir), DATA_DIR, T0_DIR, destination)

if __name__ == "__main__":
    main()

