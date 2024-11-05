# Begin working
- clone the repository
- install the requirements: `pip install -r requirements.txt`
- download the raw / prepared data, (optional models) 
- set the paths in paths.yaml (see below)

## Exemplary paths.yaml file:

    default_raw_dir: /import/sgs.scratch/hofmanja/datasets
    datasets_prepared_dir: /import/sgs.scratch/hofmanja/datasets_prepared
    modesl: /home/hofmanja/test_nn/runs
    

## Training a model:
- run main.py

    ```
    python main.py 

    optional arguments:
    --epochs
    --inputs: permeability (k), sdf (sf), gradient (g), material id (i), default is ks
    --prev_boxes: how many boxes are used as input
    --extend: how many boxes are predicted
    --num_layers: number of convLSTM layers
    --enc_conv_features: array of number of features used in the encoding convolution
    --dec_conv_features: array of number of features used in the decoding convolution
    --enc_kernel_sizes: array of kernel sizes in the encoding convolution
    --dec_kernel_sizes: array of kernel sizes in the decoding convolution
    ```

## Testing a model and visualizing results:
- run main.py:

    ```
    python main.py --case test --model PATH_TO_MODEL (after "runs/") 
    
    optional arguments (must match the model):
    --inputs: permeability (k), sdf (sf), gradient (g), material id (i), default is ks
    --prev_boxes: how many boxes are used as input
    --extend: how many boxes are predicted
    --num_layers: number of convLSTM layers
    --enc_conv_features: array of number of features used in the encoding convolution
    --dec_conv_features: array of number of features used in the decoding convolution
    --enc_kernel_sizes: array of kernel sizes in the encoding convolution
    --dec_kernel_sizes: array of kernel sizes in the decoding convolution
    --visualize: visualize the results (default `False`)
    ```

## Finding the results:
- resulting model (`model.pt`) + normalization parameters (info.yaml) used can be found in `runs/DESTINATION` with `DESTINATION` being the user defined or default name in the call of main.py
- this folder also contains visualisations if any were made during the training/testing
- prepared datasets are in datasets_prepared (paths.yaml:`datasets_prepared_dir/`)

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
