# Datasets for Learning Single Parameters instead of a whole Temperature Field

## Structure
Each file consists of one datapoint - meaning one simulation run. The files are named after the simulation run_id.
Inputs have the size of `n_input_params` x `x_dimension` x `y_dimension`. Labels, respectively the temperature fields, have the size of `x_dimension` x `y_dimension`.

Additionally I prepared two folders with reduced input and label size: InputParams contains again one file per datapoint with the parameters [`position of heat pump in x-direction`, `position of heat pump in y-direction`, `permeability value`, `pressure_gradient value`]. LearnableParams have the size of `n_learnable_params`, which are [`length of the 1-Kelvin-isoline`, `width of the 1-Kelvin-isoline`, `maximum temperature`, `location of maximum temperature`] of the two-dimensional heat plume of a simulated open loop groundwater heat pump.

The original (but too large in size) input parameters are the constant, but scaled permeability field; a pressure field defined by a constant pressure gradient (again: constant per datapoint but varying over the dataset); the position of the heat pump as a one-hot-encoded field; and the signed distance function field of the position of the heat pump. It turned out that including improves the training.

## Generation
Each raw dataset is generated with the software Pflotran and postprocessed by a python script called `prepare_dataset.py` (from https://github.com/JuliaPelzer/1HP_NN) to reduce the dimensionality to 2, store it as .pt files, apply transforms and reduce the dataset to a desired size, input parameters and input parameter time step. This gives us the folders `Inputs` and `Labels`. Afterwards the script `prepare_learnable_params.py` is called. This gives us the folder `InputParams` and `LearnableParams`.

**WARNING** Some functions are copied from the mentioned repository for easier usage. These functions are in the folder `utils_copied_from_git`.

### Step-by-step generation
1. Generate raw dataset with Pflotran.
2. the best approach to run 'prepare_dataset' in the new filestructure is to call main.py with the following arguments:
```
python main.py --dataset_raw --device --epochs 1 --inputs gksi 
```
3. remove the respective runs folder (dataset with 1 epoch is not enough)
4. run prepare_learnable_params.py with the respective paths and dataset names in the main function

## Dataset

### benchmark_dataset_2d_100datapoints_5years
This dataset consists of 100 datapoints. The labels and learnable parameters are after 5 years, not the required 27.5 years to definitely achieve a steady state solution since at 27.5 years lots of the plumes are larger than the domain and therefore the length and width are undefined. After 5 years still some datapoints have heat plumes that are larger than the domain but only about 10/100. If the heat plume is larger than the domain the values for length and width are set to NaN/null. Another case is that the background flow is so strong that the heat is transported away so quickly that there is no 1-Kelvin-isoline because the temperature delta is never above 1K/1°C. In this case length and width are set to zero.
The number of grid cells is 256x16. The cell size is 5x5 meters.

## Working with the Dataset

If you want to use this dataset for training a FCNN you can import the reduced data as follows:

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

If you want to import the data for training a CNN you can import the full data instead (Inputs, Labels-folders).

## License
The code is licensed under GNU GPLv3.0. The data is licensed under CC-BY-4.0.