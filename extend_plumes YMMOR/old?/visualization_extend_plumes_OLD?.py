import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from networks.unet import UNet

from postprocessing.visualization import _aligned_colorbar


def load_all(dataset_prep:Path, model_path:Path, run_name: str) -> (UNet, (torch.Tensor, torch.Tensor)):

    inputs = torch.load(dataset_prep / "Inputs" / f"{run_name}.pt")
    labels = torch.load(dataset_prep / "Labels" / f"{run_name}.pt")

    in_channels = inputs.shape[0] + 1
    model = UNet(in_channels=in_channels).float()
    model.load(model_path)

    return model, (inputs, labels)


def infer(model: UNet, data: torch.Tensor, box_size:int, skip_per_dir:int) -> np.ndarray:
    # inputs, labels = data
    output = torch.cat(data, dim=0).unsqueeze(0)
    # output = torch.cat((inputs_orig, labels_orig), dim=0).unsqueeze(0)
    start_prediction_box = box_size
    start_input_box = 0
    while start_prediction_box + box_size <= output.shape[2]:
        out = model(output[:, :, start_input_box:start_input_box+box_size])
        output[:,-1,start_prediction_box:start_prediction_box+box_size] = out
        start_prediction_box += skip_per_dir
        start_input_box += skip_per_dir

    return output[0,-1,:,:].detach().numpy()


def visualization(data_orig: (torch.Tensor, torch.Tensor), prediction: np.ndarray, run_name: str, destination:Path):
    inputs, labels = data_orig
    print(inputs.shape, labels.shape, prediction.shape)
    num_plots = len(inputs) + 3
    _, axes = plt.subplots(num_plots,1, sharex=True)

    for i in range(len(inputs)):
        plt.sca(axes[i])
        plt.imshow(inputs[i].T)
        plt.title(f"Input {i}")
        _aligned_colorbar()
    
    i+=1
    plt.sca(axes[i])
    plt.imshow(labels[0,:,:].T)
    plt.title("Label: Temperature")
    _aligned_colorbar()

    i+=1
    plt.sca(axes[i])
    plt.imshow(prediction.T)
    plt.title("Prediction: Temperature (still normed)")
    _aligned_colorbar()

    i+=1
    plt.sca(axes[i])
    plt.imshow((labels[0, :, :]-prediction).T)
    plt.title("Diffence between label and prediction")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(destination / f"{run_name}_extend_plumes_infer_based_throughput.png")
    # plt.show()

def pipeline_datapoint(dataset_prep:Path, model_path:Path, run_name: str, box_size:int, skip_per_dir:int, destination:Path):
    model, data = load_all(dataset_prep, model_path, run_name)
    output = infer(model, data, box_size, skip_per_dir)
    visualization(data, output, run_name, destination)

def pipeline_dataset(dataset_prep:Path, model_path:Path, args: argparse.Namespace):
    # sweep data_name folder
    for idx, file in enumerate((dataset_prep / "Inputs").iterdir()):
        run_name = file.stem.split(".")[0]
        pipeline_datapoint(dataset_prep, model_path, run_name, args.len_box, args.skip_per_dir, args.destination)

        if idx > 5:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prep", type=str, default="dataset_medium_k_3e-10_58dp inputs_gksi") #"dataset_medium_k_3e-10_1000dp inputs_gksi"
    parser.add_argument("--model", type=str, default="dataset_medium_k_3e-10_1000dp inputs_gksi case_train box256 skip128")
    parser.add_argument("--len_box", type=int, default=256)
    parser.add_argument("--skip_per_dir", type=int, default=128)
    parser.add_argument("--destination", type=str, default="")
    args = parser.parse_args()

    dataset_prep = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes") / args.data_prep
    model_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes2") / args.model
    if args.destination == "":
        args.destination = model_path / f"visu_{dataset_prep.stem}"
    args.destination.mkdir(exist_ok=True)

    pipeline_dataset(dataset_prep, model_path, args)