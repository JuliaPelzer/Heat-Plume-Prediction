# Datasets for Learning Single Parameters instead of a whole Temperature Field

## Structure
Each file consists of one datapoint - meaning one simulation run. The files are named after the simulation run_id.
Inputs have the size of `n_input_params` x `x_dimension` x `y_dimension`. Labels, respectively the temperature fields, have the size of `x_dimension` x `y_dimension`.

Additionally I prepared two folders with reduced input and label size: InputParams contains again one file per datapoint with the parameters [`position of heat pump in x-direction`, `position of heat pump in y-direction`, `permeability value`, `pressure_gradient value`]. LearnableParams have the size of `n_learnable_params`, which are [`length of the 1-Kelvin-isoline`, `width of the 1-Kelvin-isoline`, `maximum temperature`, `location of maximum temperature`] of the two-dimensional heat plume of a simulated open loop groundwater heat pump.

The original (but too large in size) input parameters are the constant, but scaled permeability field; a pressure field defined by a constant pressure gradient (again: constant per datapoint but varying over the dataset); the position of the heat pump as a one-hot-encoded field; and the signed distance function field of the position of the heat pump. It turned out that including improves the training.

## Generation
Each raw dataset is generated with the software Pflotran and postprocessed by a python script called `prepare_dataset.py` to reduce the dimensionality to 2, store it as .pt files, apply transforms and reduce the dataset to a desired size, input parameters and input parameter time step. This gives us the folders `Inputs` and `Labels`. Afterwards the script `prepare_learnable_params.py` is called. This gives us the folder `InputParams` and `LearnableParams`.

## Datasets

### benchmark_dataset_2d_100datapoints_5years
This dataset consists of 100 datapoints. The labels and learnable parameters are after 5 years, not the required 27.5 years to definitely achieve a steady state solution since at 27.5 years lots of the plumes are larger than the domain and therefor the length and width are not defined. After 5 years still some datapoints have heat plumes that are larger than the domain but only about 10/100. If the heat plume is larger than the domain the values for length and width are set to NaN/null. Another case is that the background flow is so strong that the heat is transported away so quickly that there is no 1-Kelvin-isoline. In this case length and width are set to zero.
The number of grid cells is 256x16. The cell size is 5x5 meters.

## Import Datapoints
```
import os
import torch
import pathlib

datasets_path = # whereever your datasets are stored
dataset_name = "benchmark_dataset_2d_100datapoints_5years"
dataset_path = pathlib.Path(os.path.join(datasets_path, dataset_name))

inputs = torch.load(os.path.join(dataset_path, "InputParams", "RUN_0.pt"))
labels = torch.load(os.path.join(dataset_path, "LearnableParams", "RUN_0.pt"))
```