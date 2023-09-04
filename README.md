## GPU support
- if you want to use the GPU, you need to install pflotran with cuda support
- check nvidia-smi for the available gpus and cuda version
- `export CUDA_VISIBLE_DEVICES=<gpu_id>` (e.g. 0)
- if the gpu is not found after suspension, try

    `sudo rmmod nvidia_uvm
    sudo modprobe nvidia_uvm`

    if it does not help, you have to reboot

## Python Requirements
- pyyaml
- torch
- h5py
- tensorboard
- tqdm
- pytest

## Begin working
- make a paths.yaml file in same folder as main.py. It should look like this

    ` default_raw_dir:  /scratch/sgs/pelzerja/datasets/1hp_boxes
    datasets_prepared_dir: /home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN
    datasets_prepared_dir_2hp: /home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_NN`

- have your data in the default_raw_dir OR an already prepared dataset in the datasets_prepared_dir

- run main.py

- for running a 2HP-NN you need the prepared 2HP-dataset in datasets_prepared_dir_2hp