import os
import pathlib
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN") # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  

from preprocessing.data_stuff.dataset import SimulationDataset
from utils.utils_data import load_yaml
from postprocessing.visualization import DataToVisualize


def main_learnable_params(dataset_path: str):
    # main function to be called from outside

    dataset = SimulationDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=100)

    norm = dataloader.dataset.norm
    for inputs, labels in dataloader:
        for datapoint_id in range(inputs.shape[0]):
            # get data
            x = inputs[datapoint_id]
            x = torch.unsqueeze(x, 0)
            y = labels[datapoint_id]
            domain_size_meters = (np.array(dataloader.dataset.info["CellsSize"][:2]) * x.shape[2:])

            # reverse transform for plotting real values
            x = norm.reverse(x.detach().cpu().squeeze(), "Inputs")
            y = norm.reverse(y.detach().cpu(),"Labels")[0]

            data_Temp = DataToVisualize(y, "Temperature True [°C]",domain_size_meters)
            learnable_params = calc_learnable_params(data_Temp, domain_size_meters=domain_size_meters)
        
            run_id = dataset.get_run_id(datapoint_id)
            torch.save(learnable_params, os.path.join(dataset_path, "LearnableParams", run_id))

def calc_learnable_params(datapoint: DataToVisualize, domain_size_meters: np.array = np.array([1280, 80])):
    # calc length and width of 1K - contour line

    loc_max_temp = datapoint.data.argmax()
    # loc_max_temp = torch.tensor([loc_max_temp%domain_size_meters[1], loc_max_temp//domain_size_meters[1]%domain_size_meters[0], loc_max_temp//domain_size_meters[0]//domain_size_meters[1]%5])
    max_temp = datapoint.data.max()

    CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
    
    try:
        path = CS.collections[0].get_paths()[0].vertices
        x_min = path[:,0].min()
        x_max = path[:,0].max()
        y_min = path[:,1].min()
        y_max = path[:,1].max()

        # determine len, width, include checks for out of bounds
        if y_min <= 0 or y_max >= domain_size_meters[1]:
            width = None
        else:
            width = y_max - y_min
        if x_min <= 0 or x_max >= domain_size_meters[0]:
            length = None
        else:
            length = x_max - x_min

    # special case of no contour line at all
    except IndexError:
        length, width = 0, 0

    # convert to torch tensor
    learnable_params = np.array([length, width, max_temp, loc_max_temp], dtype=np.float32)
    np.nan_to_num(learnable_params, copy=True)
    learnable_params = torch.tensor(learnable_params)
        
    return learnable_params

def main_input_params(dataset_raw_path: pathlib.Path, destination_path: pathlib.Path):
    # attention: expects the loc_hp to NOT vary over the dataset
    
    settings = load_yaml(os.path.join(dataset_raw_path, "inputs"))
    loc_hp = settings["grid"]["loc_hp"]

    for dir in dataset_raw_path.iterdir():
        if dir.is_dir() and dir.name != "inputs":
            run_id, permeability, pressure_gradient = get_input_params(dir)
            input_params = torch.tensor([loc_hp[0], loc_hp[1], permeability, pressure_gradient])
            torch.save(input_params, os.path.join(destination_path, "InputParams", run_id+".pt"))

def get_input_params(dir: pathlib.Path):
    name = dir.name
    with open(os.path.join(dir, "permeability_iso.txt")) as f:
        permeability = f.readline().split()[1]
    with open(os.path.join(dir, "pressure_gradient.txt")) as f:
        pressure_gradient = f.readline().split()[2]
    return name, float(permeability), float(pressure_gradient)

def test(dataset_path: str):
    # allows for visual verification of the learnable parameters

    params = torch.load(os.path.join(dataset_path, "LearnableParams", "RUN_0.pt"))
    print(params)
    output = torch.load(os.path.join(dataset_path, "Labels", "RUN_0.pt"))
    plt.imshow(output[0,:,:].T)
    plt.xticks([0, 256/6,256/3,256/2, 256])
    plt.show()

if __name__ == "__main__":
    # datasets_raw_path = "/home/pelzerja/Development/datasets/learn_parameters"
    datasets_raw_path = "/scratch/sgs/pelzerja/datasets/1hp_boxes"
    dataset_name = "benchmark_dataset_2d_100datapoints"
    dataset_raw_path = pathlib.Path(os.path.join(datasets_raw_path, dataset_name))

    # datasets_prepared_1hp_path = "/home/pelzerja/Development/datasets_prepared/learn_parameters"
    datasets_input_path = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN"
    dataset_input_name = f"{dataset_name}_5years"
    dataset_input_path = pathlib.Path(os.path.join(datasets_input_path, dataset_input_name))

    datasets_target_path = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/learn_parameters"
    dataset_target_path = pathlib.Path(os.path.join(datasets_target_path, dataset_input_name))
    shutil.copytree(dataset_input_path, dataset_target_path)
 
    dataset_target_path.joinpath("InputParams").mkdir(parents=True, exist_ok=True)
    main_input_params(dataset_raw_path, dataset_target_path)

    dataset_target_path.joinpath("LearnableParams").mkdir(parents=True, exist_ok=True)
    main_learnable_params(dataset_target_path)