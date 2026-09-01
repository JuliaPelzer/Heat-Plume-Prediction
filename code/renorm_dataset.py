import argparse
import sys
from pathlib import Path
import shutil
import torch
from tqdm.auto import tqdm

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import load_yaml, save_yaml

def renorm_dataset(data_path: Path, new_info_path: Path, destination: Path):
    """Load a dataset, unnorm it according to its own info.yaml, renorm it according to
    `new_info_path` and store it under `destination` (default: next to the new info.yaml).

    Handles "Inputs" and "Labels" (whichever are present) and supports all normalization
    types implemented in NormalizeTransform (Rescale, LogRescale, Standardize).
    """
    data_path, new_info_path, destination = Path(data_path), Path(new_info_path), Path(destination)
    if destination.resolve() == data_path.resolve():
        raise ValueError(f"destination '{destination}' must differ from the source dataset '{data_path}'.")
    if not new_info_path.is_file():
        raise FileNotFoundError(new_info_path)

    old_norm = NormalizeTransform(load_yaml(data_path / "info.yaml"))
    new_norm = NormalizeTransform(load_yaml(new_info_path))

    for data_type in ["Inputs", "Labels"]:
        source_dir = data_path / data_type
        if not source_dir.is_dir():
            continue
        (destination / data_type).mkdir(parents=True, exist_ok=True)
        run_names = sorted(f.name for f in source_dir.iterdir() if f.suffix == ".pt")
        for name in tqdm(run_names, desc=f"Re-normalizing {data_type}"):
            data = torch.load(source_dir / name, weights_only=False)
            # if data_type == "Inputs" and len(data) != 9: # TODO interim
            #     continue # TODO interim
            #     # TODO manually rm respective Labels
            data = new_norm(old_norm.reverse(data, data_type), data_type)
            torch.save(data, destination / data_type / name)

    for f in data_path.iterdir():  # keep additional files (e.g. temperature_injection_series.npy)
        if f.is_file() and f.name != "info.yaml":
            shutil.copy(f, destination / f.name)
    save_yaml(load_yaml(new_info_path),destination/"info.yaml")
    print(f"Stored re-normalized dataset under {destination}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renorm-data", type=str, default=None,
                            help="path of a dataset to un- and re-normalize instead of running the full pipeline")
    parser.add_argument("--new-info", type=str, default=None,
                        help="path of the info.yaml defining the target normalization; data stored next to it")
    parser.add_argument("--destination", type=str, default=None,
                        help="storage path for the re-normalized dataset")
    args = parser.parse_args()

    assert args.renorm_data and args.new_info, "--renorm-data and --new-info must be given together."
    renorm_dataset(Path(args.renorm_data), Path(args.new_info), Path(args.destination))

    print("done")


if __name__ == "__main__":
    main()
