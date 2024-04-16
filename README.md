# Begin working
- clone the repository and go to the respective branch
- install the requirements: `pip install -r requirements.txt`
- download the raw / prepared data, (optional models and data sets for 2nd stage) 
- set the paths in paths.yaml (see later)

## Exemplary paths.yaml file:

    ```
    default_raw_dir: /scratch/sgs/pelzerja/datasets # where the raw 1st stage data is stored
    datasets_prepared_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared # where the prepared 1st stage data is stored
    datasets_raw_domain_dir: /scratch/sgs/pelzerja/datasets/2hps_demonstrator_copy_of_local
    datasets_prepared_domain_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_domain
    prepared_1hp_best_models_and_data_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN_preparation_BEST_models_and_data
    models_2hp_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN/runs
    datasets_prepared_dir_2hp: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_boxes
    ```

## Training a 1st stage model (1HP-NN):
- run main.py

    ```
    python main.py --dataset_raw NAME_OF_DATASET --problem 2stages
    ```
    optional arguments:
    --inputs: make sure, they are the same as in the model (default `gksi`)
    --visualize: visualize the results (default `False`)

## Infer a 1st stage model:
- run main.py:

    ```
    python main.py --dataset_raw NAME_OF_DATASET --case test --model PATH_TO_MODEL (after "runs/") --problem 2stages
    
    optional arguments:
    --inputs: make sure, they are the same as in the model (default `gksi`)
    --visualize: visualize the results (default `False`)
    ```
## Training a 2nd stage model (2HP-NN):
- for running a 2HP-NN you need the prepared 2HP-dataset in datasets_prepared_dir_2hp (paths.yaml)
- for preparing 2HP-NN: expects that 1HP-NN exists and trained on; for 2HP-NN (including preparation) run main.py with the following arguments:

    ```
    python main.py --dataset_raw NAME_OF_DATASET --case_2hp True --inputs INPUTS (rather preparation case from 1HP-NN) --problem 2stages
    more information on required arguments:
    --inputs: make sure, they are the same as in the model (default `gksi`) + the number of datapoints -> e.g. `gksi1000`
    -- model: not required, automatically (depending on the input case) taken from paths.yaml:prepared_1hp_best_models_and_data_dir

    optional arguments:
    --visualize: visualize the results (default `False`)
    --case: `test`, `train` or `finetune` (default `train`)
    ```

## Preparing a merged 2nd stage model:
datasets are e.g. dataset_2hps_1fixed_100dp_2hp_gksi_1000dp, dataset_2hps_1fixed_10dp_2hp_gksi_1000dp, dataset_2hps_1fixed_1000dp_2hp_gksi_1000dp

```
python prepare_2ndstage.py --dataset "dataset_2hps_1fixed_10dp inputs_gki100 boxes" --merge True
```


## Infer a 2nd stage model:

- as inferring a 1st stage model but with model name from trained 2nd stage model

-- case: `test`

## Training an extend plumes model:
- train a model for the first box (e.g. 1HPNN) or via `problem extend1`, e.g.

    ```
    python3 main.py --problem extend1 --dataset_raw dataset_medium_10dp

    optional arguments:
    --device: `cuda:0` or `cuda:1` etc. or `cpu` (default `cuda:0`)
    --visualize: visualize the results (default `False`)
    --notes: string input to get a `notes.txt` file in model folder
    --epochs: number of epochs to train
    --destination: define a user specific name for the destination folder of the trained model (default build from dataset name + inputs)
    --len_box: where should the dataset be cut off (length) to only train on the first box (=length of first box)
    ```
- save prepared dataset (paths.yaml:datasets_prepared_dir / extend_plumes) with extension "extend1" to not confuse it with extend2, where full length of the dataset is required
- extend plumes idea: each simulation run is cut into several datapoints of length `len_box`. From the inputs+temperature field of the prior (lefthand) box, the temperature field of the next box is predicted.
- train a model for the extension of boxes via `problem extend2`, e.g.
    ```
    python3 main.py --problem extend2 --dataset_raw dataset_medium_10dp --inputs gk --len_box 128 --skip_per_dir 32

    arguments:
    -- inputs: tipp: exclude `s` (signed distance function of positions of heat pump) because of difficulties with arbitrary long fields but length-dependent `s` and `i` (one hot encoding of position of heat pump) 
    -- len_box: length of each box predicted and used for training (perceptive field)
    --skip_per_dir: if skip=len_box: no overlap between different datapoints. skip should never be larger than len_box, otherwise there are parts of the simulation run that are never seen in training.
    ```

## Infering and combining both models of extend plumes:
- see `extend_plumes.py:pipeline_infer_extend_plumes` and `__main__` on how to use it.

## Finding the results:
- resulting model (`model.pt`) + normalization parameters (info.yaml) used can be found in `runs/PROBLEM/DESTINATION` with `PROBLEM` in [1hpnn, 2hpnn, allin1, extend_plumes1, extend_plumes2] and `DESTINATION` being the user defined or default name in the call of main.py
- this folder also contains visualisations if any were made during the training/inference
- prepared datasets are in datasets_prepared (paths.yaml:`datasets_prepared_dir/PROBLEM`)

## Postprocessing:
- option 1: set `--visu` to True during training / inference
- option 2: run `python postprocessing/animation.py --model MODEL_NAME --data DATA_PREPARED_NAME --run RUN_ID` to get an animation of the training process. Works only if several models were saved during training.

# Logging your training progress:
- use command 
    ```
    tensorboard --logdir=runs/ --host localhost --port 8088
    ```
    in command line and follow the instructions - there you can log the progress of e.g. the loss graph, the lr-schedule, etc.
- if you want to change the learning rate during training (e.g. because you see no progress anymore) you can press strg+c to interrupt the training, enter a new lr and continue the training. The new lr is documented in the lr-schedule in `learning_rate_history.csv` in the models folder in runs/

# GPU support
- if you want to use the GPU, you need to install with cuda support
- check nvidia-smi for the available gpus and cuda version
- `export CUDA_VISIBLE_DEVICES=<gpu_id>` (e.g. 0)
- if the gpu is not found after suspension, try

    `sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm`

    if it does not help, you have to reboot

# important commits
- directly after paper submission (Oct. '23): cdc41426184756b9b1870e5c0f52d399bee0fae0
- after clean up, one month after paper submission (Oct. '23): c8da3da
- release fosr students to extend_plumes (Mar. '24): ed884f9fb3b8af9808f7abcfee9a0810e8c0fe03, branch release_24
- release for students to work on first stage (e.g. rotational equivariance) (Mar. '24): 083bb976dfccc52d1, branch release_24
- cleaned up merge of extend_plumes and allin1 (Apr. '24): ab1e389973c59, branch allin1_preprocessing
- functioning allin1 + preprocessing 1st stage + extend plumes (Apr. '24): 9a72b9a8b003e5, branch allin1_preprocessing & 
- removed "other_models"