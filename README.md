# Training, Hyperparameter Search and Evaluation of LGCNN
This repository contains code for hyperparameter optimization and training of LGCNN and a vanilla UNet model using PyTorch. It also includes functionality for evaluating the trained model and reproducing results from the associated research paper.

Pre-trained models and raw datasets are part of the supplementary material of the paper. The preparation of the datasets for the first step or the full pipeline is done automatically, when `main.py` is run and the datasets is not yet prepared or not in the correct folder (see below). For Step 3, the data has to be prepared manually, see Step 2.

## Table of Contents
1. Some important files
2. Getting started
3. Training of an LGCNN (or another model)
4. Inference of an LGCNN
5. Postprocessing, incl. evaluation of metrics

## Some important files in code/
- **`main.py`**: Contains code for training a single model (Step 1 and 3 of LGCNN, and vanilla UNet) or conducting a hyperparameter search.
- **`lgcnn_infer_until_step3.py`**: Contains code for importing datasets into .pt format, preparing it, calculating streamlines (Step 2) and preparing inputs for Step 3 (which should for benchmarkingbe evaluated with the official evaluation code in [Benchmarking Repo](https://github.com/JuliaPelzer/Stepwise-Benchmarking-HPs))
- **`default_HPS_options.yaml`**: Defines the default search ranges for hyperparameter optimization. Needs to be copied to the hyperparameter search folder and renamed to `HPS_options.yaml` to be used, also used for setting the model parameters if no hyperparameter search is performed. In that case, please make sure to set the hyperparameter values in `HPS_options.yaml` to a single value, e.g., `lr: 0.001` instead of `lr: [0.001, 0.01]`.
- **`runs/example`**: Contains example files for the command line arguments and hyperparameters. These files need to be copied to the folder where a new model is trained.

## Getting started
- install the requirements: via`pip install -r requirements.txt`
- download the preprocessed datasets of step 1, 2 r 3 (or 3b) from our [Dataverse](https://darus.uni-stuttgart.de/dataverse/heat_pumps_stepwise_benchmarking), optional: trained models
- make separate folders for the raw and prepared datasets and the models, e.g., `datasets`, `datasets_prep` and `runs` as in the example
- `cd` into the code folder

## Training of LGCNN (or another model)
Check the path variable

    PATH_MODELS_DIR = Path("../runs") # TODO: change to your models/ results directory

in `main.py`.

### Step 1 (predict v)
- make a new folder for the model in `runs/`, e.g., `NAME_OF_DIR_PREDICT_V`
- define a `command_line_argument.yaml` in `NAME_OF_DIR_PREDICT_V` acc. to the example in `example/command_line_arguments.yaml`, and `HPS_options.yaml` in `NAME_OF_DIR_PREDICT_V` acc. to the example in `example/HPS_options.yaml` (hyperparameters of your model or of the hyperparameter search)
- make sure to prepare a proper dataset with velocities as labels
- run `python main.py --destination NAME_OF_DIR_PREDICT_V`
- if you want to run a hyperparameter search, make sure to add the parameter `--hsearch True` and to give more than one value for the hyperparameters in `HPS_options.yaml`

###  Step 2 (calculate streamlines and prepare inputs for Step 3)
- run `python step2_streamlines/streamlines_main.py --data_path` with 
    - the path to the prepared input dataset for this step
    - the method is by default set to `Radau` (implicit RK method), but you can also set it to `RK23` (RungaKutta 2(3)) or `RK45` (RungaKutta 4(5))

### Step 3 (predict T)
- check that the required prepared dataset exists (download or run Step 2)
- define a `command_line_argument.yaml` in `NAME_OF_DIR_PREDICT_T` acc. to example in `example/command_line_arguments.yaml` and `HPS_options.yaml` (same as in Step 1)
- make sure that `data_pred`in `command_line_arguments.yaml` points to the prepared dataset from Step 2 with the true end-time velocities, i.e., that the path is correct and that the dataset is prepared properly (based on `_interimvelocities` dataset from our dataverse)
- run `python main.py --destination NAME_OF_DIR_PREDICT_T`
- if you want to run a hyperparameter search, make sure to add the parameter `--hsearch True` 

## Inference of LGCNN
- each model individually can be inferred by running `python main.py --destination NAME_OF_TRAINED_MODEL_DIR` when setting the parameter `case: test` in `command_line_arguments.yaml` in the respective directory
- to infer the whole pipeline with the models from our dataverse, i.e., including inputs wf, trained on residuals, etc:
    - make sure, that you have (downloaded) a trained model for Step 1 and Step 3, e.g., `NAME_OF_DIR_PREDICT_V` and `NAME_OF_DIR_PREDICT_T`
    - run `python lgcnn_infer_until_step3.py --src_npz ORIG_DATA_DIR --model_v MODEL_DIR_PREDICT_V --model_T MODEL_DIR_PREDICT_T`.
    - if you work with the data directly downloaded from our dataverse, you can set `--src_npz` to the path of the original dataset, e.g., `../datasets/step1/step1_test/`, else if you already have the prepared torch files, you can set `--src_pt` directly and leave src_npz empty, e.g., `--src_pt ../datasets_prepared/step1/step1_test/`
    - `--model_v` and `--model_T` need to point to the trained models, e.g., `../runs/NAME_OF_DIR_PREDICT_V` and `../runs/NAME_OF_DIR_PREDICT_T`.
    - the script will produce several interim datasets starting from the point where it left off. The predicted dataset for step 3 will be saved in `ORIG_DATA_DIR_wf_not_replace_v_renormed_prededV+s(t,Q,Ttrend)_renormed`.
    - Next, you can either call step 3 on this dataset as described above, or you can run the automatic evaluation, which is the same as for the leaderboard evaluation. It is stored in a different repository: [Benchmarking Repo](https://github.com/JuliaPelzer/Stepwise-Benchmarking-HPs).

## Postprocessing
- set `visu: True` during training / inference in `command_line_arguments.yaml`
- set `case: test` in `command_line_arguments.yaml` to only run inference, change the `data_prep` to whatever dataset you want to visualize, e.g., `step1_test` or `step3_test`, not necessarily the dataset used for training
