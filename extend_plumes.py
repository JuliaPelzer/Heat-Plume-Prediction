import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import yaml
import os
from typing import List
from pathlib import Path

from networks.unetVariants import *
from postprocessing.visualization import _aligned_colorbar

def load_front(model_front_path, dataset_front, run_id, model_name="model.pt"):
    # load and init first box stuff (extend1: data+model)
    inputs_front = torch.load(dataset_front / "Inputs" / f"RUN_{run_id}.pt")
    labels_front = torch.load(dataset_front / "Labels" / f"RUN_{run_id}.pt")
    model_front = UNetHalfPad(in_channels=4).float()
    model_front.load(model_front_path, model_name=model_name)
    model_front.eval()
    return model_front, inputs_front, labels_front

def load_extend(model_path, dataset_prep, run_id, visu=False, model_name:str = "model.pt"):
    # load and init extend boxes stuff (extend2: data+model)
    inputs = torch.load(dataset_prep / "Inputs" / f"RUN_{run_id}.pt")
    labels = torch.load(dataset_prep / "Labels" / f"RUN_{run_id}.pt")
    info = yaml.safe_load((model_path / "info.yaml").open("r"))
    temp_norm = info["Labels"]["Temperature [C]"]
    if visu:
        plt.figure(figsize=(15, 5))
        plt.imshow(labels[0, :, :].T)
        _aligned_colorbar()
        plt.show()

    model = UNetHalfPad2(in_channels=3).float() ## VERSION for only T - WIP: in_channels=1
    model.load(model_path, model_name=model_name)
    model.eval()
    return model, inputs, labels, temp_norm

def update_params(params, model_path, temp_norm):
    # params according to model_extend
    params["temp_norm"] = temp_norm
    with open(model_path / "command_line_arguments.yaml", "r") as file:
        data = yaml.load(file, Loader=yaml.Loader)
        params["box_size"] = data["len_box"]
        params["skip_per_dir"] = data["skip_per_dir"]
        params["inputs"] = data["inputs"]
    return params

def load_models_and_data(model_path, model_front_path, dataset_prep, dataset_front, params, run_id, visu=False, model_name:str = "model.pt", model_name_front:str = "model.pt", case:str="both"):
    model_front, inputs_front, model, inputs, labels = None, None, None, None, None

    if case in ["front", "both"]:
        model_front, inputs_front, _ = load_front(model_front_path, dataset_front, run_id, model_name=model_name_front)
    if case in ["extend", "both"]:
        model, inputs, labels, temp_norm = load_extend(model_path, dataset_prep, run_id, visu=visu, model_name=model_name)
        params = update_params(params, model_path, temp_norm)

    return model, model_front, inputs, labels, inputs_front, params

def infer(model, inputs, labels, params, first_box:bool=True, visu:bool=True, front:List=None):
    box_size = params["box_size"]
    start_input_box = params["start_input_box"]
    skip_per_dir = params["skip_per_dir"]
    skip_in_field = params["skip_in_field"]
    rm_boundary_l = params["rm_boundary_l"]
    rm_boundary_r = params["rm_boundary_r"]

    output_all = torch.cat((inputs, labels), dim=0).unsqueeze(0)
    # output_all = labels.unsqueeze(0).unsqueeze(0) ## VERSION for only T - WIP

    if front is not None:
        box_front = 128
        model_front, inputs_front = front
        output_front = model_front(inputs_front[:, :box_front].unsqueeze(0).detach())
        output_all[0,-1,:box_front] = output_front # vs output_all[:,...]

        if visu:
            fig, axes = plt.subplots(1,3, sharex=True, figsize=(15,5))
            axes[0].imshow(labels[0,:box_front].detach().numpy().T, **params["colorargs"])
            axes[0].set_title("Label")
            plt.sca(axes[0])
            _aligned_colorbar(axes[0].imshow(labels[0,:box_front].detach().numpy().T))
            axes[1].imshow(output_front[0, 0].detach().numpy().T, **params["colorargs"])
            axes[1].set_title("Prediction")
            axes[2].imshow((labels[0,:box_front]-output_front[0, 0]).detach().numpy().T)
            axes[2].set_title("Difference")
            plt.sca(axes[2])
            _aligned_colorbar(axes[2].imshow((labels[0,:box_front]-output_front[0, 0]).detach().numpy().T))
            plt.show()
        

    if not first_box and start_input_box < skip_per_dir:
        start_input_box = skip_per_dir

    # assert rm_boundary * 2 < skip_in_field
    assert skip_in_field + rm_boundary_l + rm_boundary_r <= box_size, f"not ensured that neetless predictions {skip_in_field} {rm_boundary_l} {rm_boundary_r} {box_size}"
    assert (skip_in_field%skip_per_dir) == 0, f"should be familiar with this part of a field {skip_in_field} {skip_per_dir}"
    assert (start_input_box%skip_per_dir) == 0, f"should be familiar with this part of a field {start_input_box} {skip_per_dir}"
    assert rm_boundary_r >= 1, "right boundary value should be at least 1 (that would be that nothing is removed on that side)"
    counter = 0

    while start_input_box + 2*box_size <= output_all.shape[2]:
        input_tmp = deepcopy(output_all[:, :-1, start_input_box+box_size:start_input_box+2*box_size].detach())
        input_tmp_T = deepcopy(output_all[:, -1, start_input_box+box_size-1].detach()) # axes of Temp = last
        input_tmp_T = input_tmp_T.unsqueeze(1).repeat(1, 1, box_size, 1)
        input_tmp = torch.cat((input_tmp, input_tmp_T), dim=1)

        output_tmp = model(input_tmp)

        output_all[0,2,start_input_box+box_size+rm_boundary_l : start_input_box+2*box_size-rm_boundary_r] = output_tmp[:,:,rm_boundary_l:-rm_boundary_r]

        if visu and counter < 6:
            _, axes = plt.subplots(1,5, sharex=True, figsize=(15,5))
            axes[0].imshow(input_tmp[0, 2].detach().numpy().T, **params["colorargs"])
            axes[0].set_title("Input")
            axes[1].imshow(output_tmp[0, 0].detach().numpy().T, **params["colorargs"])
            axes[1].set_title("Prediction")
            axes[2].imshow(output_tmp[0, 0,rm_boundary_l:-rm_boundary_r].detach().numpy().T, **params["colorargs"])
            axes[2].set_title("Prediction reduced")
            axes[3].imshow((output_tmp[0, 0,rm_boundary_l:-rm_boundary_r].detach()-labels[0,start_input_box+box_size+rm_boundary_l : start_input_box+2*box_size-rm_boundary_r]).T)
            axes[3].set_title("Difference prediction, label")
            # colorbar
            plt.sca(axes[3])
            _aligned_colorbar(axes[3].imshow((output_tmp[0, 0,rm_boundary_l:-rm_boundary_r].detach()-labels[0,start_input_box+box_size+rm_boundary_l : start_input_box+2*box_size-rm_boundary_r]).T))
            axes[4].imshow(labels[0,start_input_box+box_size : start_input_box+2*box_size].T, **params["colorargs"])
            axes[4].set_title("Label")
            plt.tight_layout()
            plt.show()

        start_input_box += skip_in_field
        counter += 1
    output_all = output_all[0,2,:,:].detach().numpy()
    return output_all

def set_start_curr_box(start_prior_box, params):
    return start_prior_box + params["box_size"] -params["overlap"]

def prep_params_and_data(inputs, labels, params):
    box_size = params["box_size"]
    start_prior_box = max(params["start_input_box"], params["skip_per_dir"])
    start_curr_box = set_start_curr_box(start_prior_box, params)
    skip_in_field = params["skip_in_field"]
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
        
    return box_size, start_prior_box, start_curr_box, skip_in_field, inputs, labels

def assemble_inputs(inputs, labels, start_prior_box: int, start_curr_box:int, params: dict, const_T_in:bool=False):
    input_curr = inputs[:, :, start_curr_box:start_curr_box+params["box_size"] ]
    if not const_T_in:
        input_prior_T = labels[:, :, start_prior_box:start_prior_box+params["box_size"]] # axes of Temp = last
    else:
        input_prior_T = labels[:, :, start_prior_box+params["box_size"]-1:start_prior_box+params["box_size"]].repeat(1, 1, params["box_size"], 1)
    input_all = torch.cat((input_curr, input_prior_T), dim=1)

    return input_all

def calc_actual_len_and_gap(output, params):
    # changed length through no padding, and therefor also changed starting point
    actual_len = output.shape[2]
    gap = (params["box_size"] - actual_len)//2
    assert params["skip_in_field"] < actual_len+gap, f"skip_in_field {params['skip_in_field']} should be smaller than actual_len+gap {actual_len+gap}"
    assert actual_len < params["box_size"], f"actual_len {actual_len} should be smaller than box_size {params['box_size']}"

    return actual_len, gap
        
def visualize(input_all, labels, output, start_curr_box, params, gap, actual_len):
    _, axes = plt.subplots(1,5, sharex=True, figsize=(15,5))
    axes[0].imshow(input_all[0, 2].detach().numpy().T, **params["colorargs"])
    axes[0].set_title("Input")
    axes[1].imshow(output[0, 0].detach().numpy().T, **params["colorargs"])
    axes[1].set_title("Prediction")
    axes[2].imshow(output[0, 0].detach().numpy().T, **params["colorargs"])
    axes[2].set_title("Prediction reduced")
    axes[3].imshow((output[0, 0].detach()-labels[0,start_curr_box+gap : start_curr_box+gap+actual_len]).T)
    axes[3].set_title("Difference prediction, label")
    # colorbars
    plt.sca(axes[3])
    _aligned_colorbar(axes[3].imshow((output[0, 0].detach()-labels[0,start_curr_box+gap : start_curr_box+gap+actual_len]).T))
    axes[4].imshow(labels[0,start_curr_box : start_curr_box+params["box_size"]].T, **params["colorargs"])
    axes[4].set_title("Label")
    plt.tight_layout()
    plt.savefig("test.png")
    plt.show()

def infer_nopad(model, inputs, labels, params, first_box:bool=True, visu:bool=True, front:List=None, overlap:bool=False, const_T_in:bool=False):
    # no padding, overlap, constant T_in
    params["overlap"] = 46 if overlap else 0 # TODO automate
    box_size, start_prior_box, start_curr_box, skip_in_field, inputs, labels = prep_params_and_data(inputs, labels, params)
    print(f"box_size: {box_size}, start_prior_box: {start_prior_box}, start_curr_box: {start_curr_box}, skip_in_field: {skip_in_field}, overlap: {params['overlap']}")

    if overlap:
        if skip_in_field > params["overlap"]: skip_in_field = params["overlap"]
        # TODO STIMMT DAS?

    while start_curr_box+box_size <= labels.shape[2]:
        input_all = assemble_inputs(inputs, labels, start_prior_box, start_curr_box, params, const_T_in)

        output = model(input_all)
        
        actual_len, gap = calc_actual_len_and_gap(output, params)
        # print(f"start_curr_box: {start_curr_box}, gap: {gap}, actual_len: {actual_len}")
        labels[0,0,start_curr_box+gap : start_curr_box+gap+actual_len] = output

        if visu: visualize(input_all, labels, output, start_curr_box, params, gap, actual_len)

        start_prior_box += skip_in_field
        start_curr_box = set_start_curr_box(start_prior_box, params)
    return labels[0,0,:,:].detach().numpy()

def rescale_temp(data, norm_info):
    # repetition of transform Rescale 
    out_min = 0
    out_max = 1
    delta = norm_info["max"] - norm_info["min"]
    return (data - out_min) / (out_max - out_min) * delta + norm_info["min"]

# def visu_rescaled_dp(output_all, labels, params, plot_name=None):
#     label_rescaled = rescale_temp(deepcopy(labels[0, :, :].numpy()), params["temp_norm"])
#     output_lessskip2_rescaled = rescale_temp(deepcopy(output_all), params["temp_norm"])

#     _, axes = plt.subplots(3,1, sharex=True, figsize=(25, 4))
#     plt.sca(axes[0])
#     plt.imshow(label_rescaled[params["start_visu"]:params["end_visu"]].T, **params["colorargs"])
#     plt.title("Label: Temperature [°C]")
#     plt.ylabel("y [cells]")
#     _aligned_colorbar()

#     plt.sca(axes[1])
#     plt.imshow(output_lessskip2_rescaled[params["start_visu"]:params["end_visu"]].T, **params["colorargs"])
#     plt.title("Prediction: Temperature [°C]")
#     plt.ylabel("y [cells]")
#     _aligned_colorbar()

#     plt.sca(axes[2])
#     plt.imshow((output_lessskip2_rescaled - label_rescaled)[params["start_visu"]:params["end_visu"]].T, **params["colorargs"])
#     plt.title("Difference: Prediction - Label [°C]")
#     plt.xlabel("x [cells]")
#     plt.ylabel("y [cells]")
#     _aligned_colorbar()

#     plt.tight_layout()
#     if plot_name:
#         plt.savefig(f"runs/extend_plumes2/results/{plot_name}.png", dpi=500)
#     plt.show()

def produce_front_comparison_pic():
    # Parameters
    params = {  "colorargs" : {"cmap": "RdBu_r"}, #"vmin":0,"vmax":1, 
                "start_visu" : 0,
                "end_visu" : 1000,
                "start_input_box" : 0, #64,
                "skip_in_field" : 64, #32,
                "rm_boundary_l" : 16,
                "rm_boundary_r" : int(16/2),}

    dataset_prep = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gk extend2")
    model_path_nfb = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes2/TESTS_inputs box128 skip64/dataset_medium_k_3e-10_1000dp inputs_gk case_train box128 skip64 noFirstBox e500 2ndRound")
    model_front_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes1/dataset_medium_64_256_gksi_1000dp_for_lukas")
    dataset_front = Path("/scratch/sgs/pelzerja/datasets_prepared/extend_plumes/dataset_long_k_3e-10_1dp inputs_gksi extend1")
    run_id = 11
    
    model, model_front, inputs, labels, inputs_front, params = load_models_and_data(model_path_nfb, model_front_path, dataset_prep, dataset_front, params, run_id=run_id, visu=False)

    front=[model_front, inputs_front]
    output_front = infer(model, inputs, labels, params, first_box=False, visu=False, front=front)

    front=None
    output_nofront = infer(model, inputs, labels, params, first_box=False, visu=False, front=front)
    vis_start = 30
    _, axes = plt.subplots(6,1, sharex=True, figsize=(16, 8))
    plt.sca(axes[0])
    label_rescaled = rescale_temp(deepcopy(labels[0, :, :].numpy()), params["temp_norm"])
    plt.imshow(label_rescaled[vis_start:1000].T, cmap="RdBu_r")
    plt.title("Label [°C]")
    plt.ylabel("y [cells]")
    _aligned_colorbar()
    plt.sca(axes[1])
    output_front_resc = rescale_temp(deepcopy(output_front), params["temp_norm"])
    plt.imshow(output_front_resc[vis_start:1000].T, cmap="RdBu_r")
    plt.title("Prediction, both steps [°C]")
    plt.ylabel("y [cells]")
    _aligned_colorbar()
    plt.sca(axes[2])
    output_nofront_resc = rescale_temp(deepcopy(output_nofront), params["temp_norm"])
    plt.imshow(output_nofront_resc[vis_start:1000].T, cmap="RdBu_r")
    plt.title("Prediction, only 2nd step [°C]")
    plt.ylabel("y [cells]")
    _aligned_colorbar()
    plt.sca(axes[3])
    plt.imshow((label_rescaled-output_front_resc)[vis_start:1000].T, cmap="RdBu_r")
    plt.title("Difference label to prediction, both steps [°C]")
    plt.ylabel("y [cells]")
    _aligned_colorbar()
    plt.sca(axes[4])
    plt.imshow((label_rescaled-output_nofront_resc)[vis_start:1000].T, cmap="RdBu_r")
    plt.title("Difference label to prediction, only 2nd step [°C]")
    plt.ylabel("y [cells]")
    _aligned_colorbar()
    plt.sca(axes[5])
    plt.imshow((output_front_resc-output_nofront_resc)[vis_start:1000].T, cmap="RdBu_r")
    plt.title("Difference: both steps - only 2nd step [°C]")
    plt.ylabel("y [cells]")
    plt.xlabel("x [cells]")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(f"extend_plumes/model_2ndRound_run{run_id}_frontComparison.png", dpi=500)
    plt.show()
