import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

sys.path.append("/home/pelzerja/Development/code_NN/")

from extend_plumes.extend_plumes import (infer, infer_nopad, load_extend,
                                         load_front, load_models_and_data,
                                         rescale_temp)

FINAL_EPOCH_ID = 9999999

def animate(rescale_temp:Callable, run_id:int, epochs:List[int], models, datasets, case:str="front", anim_name:str="model_evolution", params:dict={}):
    _, _, _, labels, _, params = load_models_and_data(models[1], models[0], datasets[1], datasets[0], params, run_id=run_id, visu=False, model_name="best_model_e0.pt", model_name_front="model.pt", case=case)
    if case == "extend":
        ref_model = models[1]
        params["len_box1"] = params["box_size"]
    else:
        ref_model = models[0]
        info = yaml.safe_load((ref_model / "info.yaml").open("r"))
        with open(ref_model / "command_line_arguments.yaml", "r") as file:
            params["len_box1"] = int(yaml.load(file, Loader=yaml.Loader)["len_box"])
        params["temp_norm"] = info["Labels"]["Temperature [C]"]
    params["colorargs_error"] = {"cmap": "RdBu_r", "vmin":-0.5,"vmax": 0.5}

    if case=="front":
        labels = labels[:,:params["len_box1"]]
    elif case in ["both", "extend"]:
        labels = labels[:, :params["end_visu"]]
    label_rescaled = rescale_temp(deepcopy(labels[0].numpy()), params["temp_norm"])
    params["colorargs"] = {"cmap": "RdBu_r", "vmin":np.min(label_rescaled),"vmax":np.max(label_rescaled)}

    if case in ["front", "extend"]:
        fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True, figsize=(10, 10))
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True, figsize=(30, 9))
    ax0.set_ylabel("y [cells]")
    ax1.set_ylabel("y [cells]")
    ax2.set_ylabel("y [cells]")
    ax2.set_xlabel("x [cells]")
    cax0 = make_axes_locatable(ax0).append_axes("right", size=0.3, pad=0.05)
    cax1 = make_axes_locatable(ax1).append_axes("right", size=0.3, pad=0.05)
    cax2 = make_axes_locatable(ax2).append_axes("right", size=0.3, pad=0.05)

    ax0.set_title(f"Label: Temperature [°C]")
    ax0.imshow(label_rescaled.squeeze().T, **params["colorargs"])
    plt.colorbar(ax0.imshow(label_rescaled.squeeze().T, **params["colorargs"]), cax=cax0)

    def update(epoch):
        _, output = update_extend(models, datasets, case, run_id, epoch, params)
        ax1.set_title(f"Output: Temperature [°C] at epoch {epoch}")
        ax1.imshow(output.squeeze().T, **params["colorargs"])
        plt.colorbar(ax1.imshow(output.squeeze().T, **params["colorargs"]), cax=cax1)
        ax2.set_title(f"Difference: Temperature [°C] at epoch {epoch}")
        ax2.imshow((output - label_rescaled).squeeze().T, **params["colorargs_error"]) # TODO
        plt.colorbar(ax2.imshow((output - label_rescaled).squeeze().T, **params["colorargs_error"]), cax=cax2)

    ani = animation.FuncAnimation(fig, update, frames=tqdm(epochs, desc="epochs"), interval=1000, repeat=True)
    ani.save(ref_model / f"run{run_id}_{anim_name}.gif", writer="pillow", fps=3)
    fig.savefig(ref_model / f"run{run_id}_{anim_name}.png")
    print(f"Animation saved as {ref_model}/run{run_id}_{anim_name}.gif")

def update_extend(models_paths, datasets_paths, case, run_id, epoch, params):
    if epoch == FINAL_EPOCH_ID:
        model_name = "model.pt"
    else:
        model_name = f"best_model_e{epoch}.pt"
        # model_name = f"interim_model_e{epoch}.pt"

    if case == "front":
        model_front = models_paths[0]
        dataset_front = datasets_paths[0]
        model, inputs, labels = load_front(model_front, dataset_front, run_id, model_name)
        inputs = inputs[:, :params["len_box1"]]
        output = model(inputs.unsqueeze(0)).detach().numpy()
    elif case == "extend":
        model = models_paths[1]
        dataset = datasets_paths[1]
        model, model_front, inputs, labels, _, params = load_models_and_data(model, models_paths[1], dataset, datasets_paths[1], params, run_id=run_id, visu=False, model_name=model_name, case=case)
        output = infer_nopad(model, inputs, labels, params, overlap=False)

        output = output[:params["end_visu"]]
    elif case == "both":
        model_front_path, model_back_path = models_paths
        dataset_front_path, dataset_back_path = datasets_paths

        model, model_front, inputs, labels, inputs_front, params = load_models_and_data(model_back_path, model_front_path, dataset_back_path, dataset_front_path, params, run_id=run_id, visu=False, model_name=model_name, model_name_front="model.pt") # not varying first box model but using the best
        output = infer(model, inputs, labels, params, first_box=False, visu=False, front=None) #[model_front, inputs_front]) # TODO include front
        output = output[:params["end_visu"]]

    output_rescaled = rescale_temp(deepcopy(output), params["temp_norm"])

    return labels, output_rescaled

def animate_extend_front():
    run_id = 0

    dataset_front = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gksi extend1")
    model_front_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes1/extend1_1000dp copy") #dataset_medium_64_256_gksi_1000dp_v2")

    pathlist = Path(model_front_path).glob('**/*.pt')
    epochs = [get_epoch(elem.name) for elem in pathlist]
    epochs.sort()
    animate(rescale_temp=rescale_temp, run_id=run_id, epochs=epochs, datasets=[dataset_front], models=[model_front_path], case="front", anim_name="anim_best_extend1_3fps")

def animate_extend_both(run_id:int, case:str="both", epochs:List[int]=None, dataset_front:Path=None, model_front:Path=None, dataset_extend:Path=None, model_extend:Path=None):
    if epochs is None:
        epochs = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300] #,400,500,600,700,800,900,1000,2000, 5000] #,3000,4000]

    params = {"colorargs" : {"cmap": "RdBu_r"},
                "start_visu" : 0,
                "end_visu" : 2000,
                "start_input_box" : 128, #64, # TODO error in handling this??
                "skip_in_field" : 32, #64,
                "rm_boundary_l" : 0, #16,
                "rm_boundary_r" : 1, #int(16/2),
                }
    if dataset_front is None:
        dataset_front = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gksi extend1")
    if model_front is None:
        model_front = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes1/dataset_medium_64_256_gksi_1000dp_v2")
    if dataset_extend is None:
        dataset_extend = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gk extend2")
    if model_extend is None:
        model_extend = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes2/extend2_1000dp") #TEST_inputs box128 skip64/dataset_medium_k_3e-10_1000dp inputs_gk case_train box128 skip64 noFirstBox e500 2ndRound")


    animate(rescale_temp=rescale_temp, run_id=run_id, epochs=epochs, datasets=[dataset_front, dataset_extend], models=[model_front, model_extend], case=case, anim_name="anim_extend2", params=params)

def get_epoch(model_name):
    if "best_model_e" in model_name:
        return int(model_name.split("_")[2][1:].split(".")[0])
    else:
        return FINAL_EPOCH_ID

def get_all_epochs(dir:Path):
    pathlist = dir.glob('**/*.pt')
    return [get_epoch(elem.name) for elem in pathlist]

def dummy_animate(model_path, data_path, run_id, epochs):
    for epoch in epochs:
        model, inputs, labels , temp_norm = load_extend(model_path, data_path, run_id, "best_model_e0.pt")
        output = model(inputs.unsqueeze(0)).detach().numpy()
        output_rescaled = rescale_temp(deepcopy(output), 0)
        fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(10, 10))
        ax0.set_ylabel("y [cells]")
        ax1.set_ylabel("y [cells]")
        ax1.set_xlabel("x [cells]")
        cax0 = make_axes_locatable(ax0).append_axes("right", size=0.3, pad=0.05)
        cax1 = make_axes_locatable(ax1).append_axes("right", size=0.3, pad=0.05)

        ax0.set_title(f"Label: Temperature [°C]")
        ax0.imshow(labels.squeeze().T, cmap="RdBu_r")
        plt.colorbar(ax0.imshow(labels.squeeze().T, cmap="RdBu_r"), cax=cax0)

        ax1.set_title(f"Output: Temperature [°C]")
        ax1.imshow(output_rescaled.squeeze().T, cmap="RdBu_r")
        plt.colorbar(ax1.imshow(output_rescaled.squeeze().T, cmap="RdBu_r"), cax=cax1)
    plt.show()
               

if __name__ == "__main__":
    datasets_path = Path("/scratch/sgs/pelzerja/datasets_prepared")
    dataset_extend_path = datasets_path / "extend_plumes"
    dataset_front_path = datasets_path / "1hp_boxes"
    models_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs")
    model_extend_path = models_path / "extend_plumes"
    model_front_path = models_path / "1hpnn"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_extend_prep", type=str, default="dataset_medium_k_3e-10_1000dp inputs_gk")
    parser.add_argument("--model_extend_prep", type=str, default="dataset_medium-10dp inputs_gk case_train box256 skip2")
    parser.add_argument("--run_id", type=int, default=5)
    parser.add_argument("--case", type=str, choices=["both", "front", "extend"], default="extend")
    parser.add_argument("--data_front_prep", type=str, default=None)
    parser.add_argument("--model_front_prep", type=str, default=None)
    args = parser.parse_args()

    dataset_extend_path = dataset_extend_path / args.data_extend_prep
    model_extend_path = model_extend_path / args.model_extend_prep
    if args.data_front_prep is not None:
        dataset_front = dataset_front_path / args.data_front_prep
    if args.model_front_prep is not None:
        model_front = model_front_path / args.model_front_prep
    assert args.case in ["extend"], f"Invalid case / not tested case: {args.case}"

    epochs = get_all_epochs(model_extend_path)
    epochs.sort()

    animate_extend_both(args.run_id, epochs=epochs, case=args.case, model_extend=model_extend_path, dataset_extend=dataset_extend_path, dataset_front=dataset_front, model_front=model_front)