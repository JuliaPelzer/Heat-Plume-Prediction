# Python Requirements
- pyyaml
- torch
- h5py
- tensorboard
- tqdm
- pytest

# Begin working
- clone the repository
- install the requirements
- download the raw / prepared datasets, models and set the paths in paths.yaml (see later)

## Infer a 1st stage model:
- run main.py:

    ```
    python main.py --dataset NAME_OF_DATASET --model PATH_TO_MODEL (after "runs/")
    
    optional arguments:
    --inputs: make sure, they are the same as in the model (default `gksi`)
    --visualize: visualize the results (default `False`)
    ```
## Exemplary paths.yaml file:

    ```
    default_raw_dir: /scratch/sgs/pelzerja/datasets/1hp_boxes
    datasets_prepared_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_boxes
    datasets_raw_domain_dir: /scratch/sgs/pelzerja/datasets/2hps_demonstrator_copy_of_local
    datasets_prepared_domain_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_domain
    prepared_1hp_best_models_and_data_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN_preparation_BEST_models_and_data
    models_2hp_dir: /home/pelzerja/pelzerja/test_nn/1HP_NN/runs
    datasets_prepared_dir_2hp: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_boxes

    ```

# GPU support
- if you want to use the GPU, you need to install pflotran with cuda support
- check nvidia-smi for the available gpus and cuda version
- `export CUDA_VISIBLE_DEVICES=<gpu_id>` (e.g. 0)
- if the gpu is not found after suspension, try

    `sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm`

    if it does not help, you have to reboot
