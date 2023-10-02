import argparse
import datetime as dt
import logging
import os
import pathlib
import time
from tqdm.auto import tqdm
import yaml
import torch
from networks.unet import UNet

from data_stuff.utils import SettingsTraining, SettingsPrepare
from prepare_dataset import prepare_dataset
from utils.utils import set_paths
from main import set_paths

def run(settings: SettingsTraining):
    timestamp_begin = time.ctime()

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.warning(f"Using {settings.device} device")

    avg_inference_times = 0
    avg_load_inference_times = 0
    num_dp = 0
    path_inputs = os.path.join(settings.datasets_path, settings.dataset_name, "Inputs")
    # for datapoint in tqdm(os.listdir(path_inputs), desc="Loading and inferring"):
    for datapoint in os.listdir(path_inputs):
        _, time_inference, time_load_inference = load_and_infer(settings, datapoint, settings.path_to_model, settings.device)
        avg_inference_times += time_inference
        avg_load_inference_times += time_load_inference
        #include visualisation in timing?
        num_dp += 1
    
    avg_inference_times = avg_inference_times / num_dp
    avg_load_inference_times = avg_load_inference_times / num_dp
    # save measurements
    with open(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, f"measurements_{settings.case}.yaml"), "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"model: {settings.model_choice}\n")
        f.write(f"number of input-channels: {len(settings.inputs_prep)}\n")
        f.write(f"input params: {settings.inputs_prep}\n")
        f.write(f"dataset location: {settings.datasets_path}\n")
        f.write(f"dataset name: {settings.dataset_name}\n")
        f.write(f"number of datapoints: {len(os.listdir(path_inputs))}\n")
        f.write(f"name_destination_folder: {settings.name_folder_destination}\n")
        f.write(f"number epochs: {settings.epochs}\n")
        f.write(f"avg loading+inference times in seconds: {avg_load_inference_times}\n")
        f.write(f"avg inference times in seconds: {avg_inference_times}\n")
        f.write(f"device: {settings.device}\n")

def load_and_infer(settings: SettingsTraining, datapoint_name:str, model_path: str, device: str = "cuda:3"):
    start_time = time.perf_counter()
    datapoint = torch.load(os.path.join(settings.datasets_path, settings.dataset_name, "Inputs", datapoint_name))
    datapoint = torch.unsqueeze(torch.Tensor(datapoint), 0).to(device)

    model = UNet(in_channels=len(settings.inputs_prep), out_channels=1, depth=3, kernel_size=5).float()
    model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=torch.device(device)))
    model.to(device)
    start_inference = time.perf_counter()
    y_out = model(datapoint).to(device)
    time_load_inference = time.perf_counter() - start_time
    time_inference = time.perf_counter() - start_inference
    return y_out, time_inference, time_load_inference

if __name__ == "__main__":
    # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--case", type=str, default="experiments")
    parser.add_argument("--path_to_model", type=str, default="current_unet_benchmark_dataset_2d_100datapoints_p_v10")
    parser.add_argument("--model_choice", type=str, default="unet")
    parser.add_argument("--inputs_prep", type=str, default="pksi")
    parser.add_argument("--name_extension", type=str, default="")
    parser.add_argument("--case_2hp", type=bool, default=False)
    args = parser.parse_args()

    args.name_folder_destination = "experiments"+args.path_to_model[7:]
    
    default_raw_dir, datasets_prepared_dir, dataset_prepared_full_path = set_paths(args.dataset_name, args.inputs_prep, args.name_extension, args.case_2hp)
    args.datasets_path = datasets_prepared_dir

    # prepare dataset if not done yet
    if not os.path.exists(dataset_prepared_full_path):
        if args.case_2hp:
            raise NotImplementedError("dataset needs to be prepared manually for 2HP case")
        else:
            args_prep = {"raw_dir": default_raw_dir,
                "datasets_dir": datasets_prepared_dir,
                "dataset_name": args.dataset_name,
                "inputs_prep": args.inputs_prep,
                "name_extension": args.name_extension}
            args_prep = SettingsPrepare(**args_prep)
            prepare_dataset(args=args_prep)
            print(f"Dataset {dataset_prepared_full_path} prepared")
    else:
        print(f"Dataset {dataset_prepared_full_path} already prepared")

    if not args.case_2hp:
        args.dataset_name += "_"+args.inputs_prep + args.name_extension

    settings = SettingsTraining(**vars(args))
    if settings.name_folder_destination == "":
        settings.name_folder_destination = f"current_{settings.model_choice}_{settings.dataset_name}"
    destination_dir = pathlib.Path(os.getcwd(), "runs", settings.name_folder_destination)
    destination_dir.mkdir(parents=True, exist_ok=True)

    settings.save()
    run(settings)