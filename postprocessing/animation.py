from copy import deepcopy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pathlib import Path
from typing import Callable, List
import yaml

from extend_plumes import rescale_temp, load_front, load_model_and_data, infer

def animate(rescale_temp:Callable, run_id:int, epochs:List[int], models, datasets, case:str="front", anim_name:str="model_evolution", params:dict={}):

    if case == "front":
        _, _, labels = load_front(models[0], datasets[0], run_id, "model.pt")
    else:
        _, _, _, labels, _, _ = load_model_and_data(models[1], models[0], datasets[1], datasets[0], params, run_id=run_id, visu=False, model_name="best_model_e0.pt", model_name_front="model.pt")
    info = yaml.safe_load((models[0] / "info.yaml").open("r"))
    with open(models[0] / "command_line_arguments.yaml", "r") as file:
        len_box = yaml.load(file, Loader=yaml.Loader)["len_box"]
    params["temp_norm"] = info["Labels"]["Temperature [C]"]
    params["len_box1"] = len_box
    params["colorargs_error"] = {"cmap": "RdBu_r", "vmin":-1,"vmax":1} # -0.5 0.5
    if case=="front":
        labels = labels[:,:params["len_box1"]]
    else:
        labels = labels[:, :params["end_visu"]]
    label_rescaled = rescale_temp(deepcopy(labels[0].numpy()), params["temp_norm"])
    params["colorargs"] = {"cmap": "RdBu_r", "vmin":np.min(label_rescaled),"vmax":np.max(label_rescaled)}

    if case == "front":
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

    def update(frame):
        epoch = epochs[frame]
        _, output = update_extend(models, datasets, case, run_id, epoch, params)
        ax1.set_title(f"Output: Temperature [°C] at epoch {epoch}")
        ax1.imshow(output.squeeze().T, **params["colorargs"])
        plt.colorbar(ax1.imshow(output.squeeze().T, **params["colorargs"]), cax=cax1)
        ax2.set_title(f"Difference: Temperature [°C] at epoch {epoch}")
        ax2.imshow((output - label_rescaled).squeeze().T, **params["colorargs_error"]) # TODO
        plt.colorbar(ax2.imshow((output - label_rescaled).squeeze().T, **params["colorargs_error"]), cax=cax2)

    ani = animation.FuncAnimation(fig, update, frames=len(epochs), interval=1000, repeat=True)
    ani.save(f"{anim_name}.gif", writer="pillow", fps=3)
    print(f"Animation saved as {anim_name}.gif")

def update_extend(models_paths, datasets_paths, case, run_id, epoch, params):
    # model_name = f"best_model_e{epoch}.pt"
    model_name = f"interim_model_e{epoch}.pt"

    if case == "both":
        model_front_path, model_back_path = models_paths
        dataset_front_path, dataset_back_path = datasets_paths

        model, model_front, inputs, labels, inputs_front, params = load_model_and_data(model_back_path, model_front_path, dataset_back_path, dataset_front_path, params, run_id=run_id, visu=False, model_name=model_name, model_name_front="model.pt") # not varying first box model but using the best
        output = infer(model, inputs, labels, params, first_box=False, visu=False, front=None) #[model_front, inputs_front]) # TODO include front
        output = output[:params["end_visu"]]

    elif case == "front":
        model_front = models_paths[0]
        dataset_front = datasets_paths[0]
        model, inputs, labels = load_front(model_front, dataset_front, run_id, model_name)
        inputs = inputs[:, :params["len_box1"]]
        output = model(inputs.unsqueeze(0)).detach().numpy()

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

def animate_extend_both():
    run_id = 8
    epochs = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300] #,400,500,600,700,800,900,1000,2000, 5000] #,3000,4000]

    params = {"colorargs" : {"cmap": "RdBu_r"},
                "start_visu" : 0,
                "end_visu" : 1000,
                "start_input_box" : 128, #64, # TODO error in handling this??
                "skip_in_field" : 64, #32,
                "rm_boundary_l" : 16,
                "rm_boundary_r" : int(16/2),}
    dataset_prep = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gk extend2")
    model_path_nfb = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes2/extend2_1000dp") #TEST_inputs box128 skip64/dataset_medium_k_3e-10_1000dp inputs_gk case_train box128 skip64 noFirstBox e500 2ndRound")

    dataset_front = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gksi extend1")
    model_front_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes1/dataset_medium_64_256_gksi_1000dp_v2")

    animate(rescale_temp=rescale_temp, run_id=run_id, epochs=epochs, datasets=[dataset_front, dataset_prep], models=[model_front_path, model_path_nfb], case="both", anim_name="anim_extend12", params=params)

def get_epoch(model_name):
    return int(model_name.split("_")[2][1:].split(".")[0])
               

if __name__ == "__main__":
    # animate_extend_front()
    animate_extend_both()