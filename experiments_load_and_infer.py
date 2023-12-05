import argparse
import logging
import os
import pathlib
import time
import torch
from networks.unet import UNet

from data_stuff.utils import SettingsTraining
from preprocessing.prepare import prepare_data_and_paths

def run_experiments(settings: SettingsTraining, measure_infer: bool = True, save: bool = True):
    timestamp_begin = time.ctime()

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.warning(f"Using {settings.device} device")

    if measure_infer:
        avg_inference_times, avg_load_inference_times, num_dp = [0, 0, 0]
    if save:
        (settings.dataset_prep / "Outputs").mkdir(parents=True, exist_ok=True)

    path_inputs = settings.dataset_prep / "Inputs"
    # for datapoint in tqdm(os.listdir(path_inputs), desc="Loading and inferring"):
    for datapoint in path_inputs.iterdir():
        y_out, time_inference, time_load_inference = load_and_infer(settings, datapoint, settings.model, settings.device, measure_infer)

        if measure_infer: 
            avg_inference_times += time_inference
            avg_load_inference_times += time_load_inference
            #include visualisation in timing?
            num_dp += 1
        
        if save:
            torch.save(y_out, settings.dataset_prep / "Outputs" / datapoint.name)
    
    if measure_infer:
        avg_inference_times = avg_inference_times / num_dp
        avg_load_inference_times = avg_load_inference_times / num_dp
        # save measurements
        with open(os.path.join(os.getcwd(), "runs", settings.destination, f"measurements_{settings.case}.yaml"), "w") as file:
            file.write(f"timestamp of beginning: {timestamp_begin}\n")
            file.write(f"timestamp of end: {time.ctime()}\n")
            file.write(f"number of input-channels: {len(settings.inputs)}\n")
            file.write(f"input params: {settings.inputs}\n")
            file.write(f"dataset name: {settings.dataset_raw}\n")
            file.write(f"number of datapoints: {len(os.listdir(path_inputs))}\n")
            file.write(f"name_destination_folder: {settings.destination}\n")
            file.write(f"number epochs: {settings.epochs}\n")
            file.write(f"avg loading+inference times in seconds: {avg_load_inference_times}\n")
            file.write(f"avg inference times in seconds: {avg_inference_times}\n")
            file.write(f"device: {settings.device}\n")

def load_and_infer(settings: SettingsTraining, datapoint_name:str, model_path: str, device: str = "cuda:3", measure_infer: bool = True):
    start_time = time.perf_counter()
    datapoint = torch.load(datapoint_name).to(device)
    datapoint = torch.unsqueeze(torch.Tensor(datapoint), 0)

    model = UNet(in_channels=len(settings.inputs)).float()
    model.load_state_dict(torch.load(model_path / "model.pt"))
    model.to(device)

    start_inference = time.perf_counter() if measure_infer else None
    y_out = model(datapoint).to(device)

    y_out = torch.squeeze(y_out, 0)

    if measure_infer:
        time_load_inference = time.perf_counter() - start_time
        time_inference = time.perf_counter() - start_inference
        return y_out, time_inference, time_load_inference
    
    return y_out, None, None 

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2d_small_1000dp", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--case", type=str, default="experiments")
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, choices=["gki", "gksi", "pksi", "gks", "gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100"], default="gksi")
    parser.add_argument("--case_2hp", type=bool, default=False)
    settings = SettingsTraining(**vars(parser.parse_args()))

    settings = prepare_data_and_paths(settings)
    
    run_experiments(settings)