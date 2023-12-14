from copy import deepcopy
import yaml
from pathlib import Path
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt

from postprocessing.visualization import _aligned_colorbar
from networks.unet import UNet, UNetBC
from data_stuff.utils import SettingsTraining
from preprocessing.prepare import prepare_data_and_paths

def cut_dataset_in_pieces(number_boxes: int, prepared1_dir, prepared_pieces_dir):
    """Cut dataset into x boxes."""
    for box_id in range(number_boxes):
        (prepared_pieces_dir / f"Inputs Box {box_id}").mkdir(parents=True, exist_ok=True)
        (prepared_pieces_dir / f"Label Box {box_id}").mkdir(parents=True, exist_ok=True)
    shutil.copy(prepared1_dir / "info.yaml", prepared_pieces_dir / "info.yaml")

    for datapoint in zip((prepared1_dir / "Inputs").iterdir(), (prepared1_dir / "Labels").iterdir()):
        input = torch.load(datapoint[0])
        label = torch.load(datapoint[1])
        name = datapoint[0].stem

        input_boxes = []
        label_boxes = []
        for i in range(number_boxes):
            len_box = input.shape[1] // number_boxes
            input_boxes.append(input[:, i * len_box : (i + 1) * len_box, :])
            label_boxes.append(label[:, i * len_box : (i + 1) * len_box, :])


        for i in range(number_boxes):
            torch.save(input_boxes[i], prepared_pieces_dir / f"Inputs Box {i}" / f"{name}.pt",)
            torch.save(label_boxes[i], prepared_pieces_dir / f"Label Box {i}" / f"{name}.pt",)

def prepare_dataset_for_2levels(dataset_name: str, number_boxes: int, input_props: str, paths: dict, prepared1_dir, prepared_pieces_dir):
    """Store boxes for 2 levels in 2 datasets."""
    revert_order = False

    # prepare 1st level
    prepared_dir_1stlevel = Path(paths["datasets_prepared_dir"]) / f"{dataset_name} cut_{number_boxes}pieces separate_boxes 1st level"
    prepared_dir_1stlevel.mkdir(parents=True, exist_ok=True)

    shutil.copy(prepared_pieces_dir / "info.yaml", prepared_dir_1stlevel / "info.yaml")
    shutil.copytree(prepared_pieces_dir / "Inputs Box 0", prepared_dir_1stlevel / "Inputs")
    shutil.copytree(prepared_pieces_dir / "Label Box 0", prepared_dir_1stlevel / "Labels")

    # prepare 2nd level
    prepared_dir_2ndlevel = Path(paths["datasets_prepared_dir"]) / f"{dataset_name} cut_{number_boxes}pieces separate_boxes 2nd level {input_props} without_last"
    prepared_dir_2ndlevel.mkdir(parents=True, exist_ok=True)
    (prepared_dir_2ndlevel / "Inputs").mkdir(parents=True, exist_ok=True)
    (prepared_dir_2ndlevel / "Labels").mkdir(parents=True, exist_ok=True)

    info = yaml.safe_load(open(prepared1_dir / "info.yaml", "r"))
    if input_props == "gkt":
        info_g    = deepcopy(info["Inputs"]["Pressure Gradient [-]"])
        info_k    = deepcopy(info["Inputs"]["Permeability X [m^2]"])
        info["Inputs"] = deepcopy(info["Labels"])
        info["Inputs"]["Pressure Gradient [-]"] = info_g
        info["Inputs"]["Permeability X [m^2]"] = info_k
        info["Inputs"]["Temperature [C]"]["index"] = 2
        # assert indices of inputs double
        idx_g = info["Inputs"]["Pressure Gradient [-]"]["index"]
        idx_k = info["Inputs"]["Permeability X [m^2]"]["index"]
        idx_t = info["Inputs"]["Temperature [C]"]["index"]
        assert  idx_g != idx_k, "indices of inputs double"
        assert  idx_g != idx_t, "indices of inputs double"
        assert  idx_k != idx_t, "indices of inputs double"
    elif input_props == "t":
        info["Inputs"] = deepcopy(info["Labels"])
        info["Inputs"]["Temperature [C]"]["index"] = 0

    yaml.safe_dump(info, open(prepared_dir_2ndlevel / "info.yaml", "w"))

    for box_id in range(number_boxes-2):
        for file_in_temp in (prepared_pieces_dir / f"Label Box {box_id}").iterdir():
            file_id = int(file_in_temp.stem.split("_")[1])
            new_id = file_id + box_id * 1000
            temp_in = torch.load(file_in_temp)
            file_inputs = prepared_pieces_dir / f"Inputs Box {box_id}" / f"RUN_{file_id}.pt"
            if input_props == "gkt":
                g_in = torch.load(file_inputs)[idx_g]
                k_in = torch.load(file_inputs)[idx_k]
                inputs = torch.zeros([3, *g_in.shape])
                if revert_order:
                    inputs[idx_g] = torch.flip(g_in, dims=(1,))
                    inputs[idx_k] = torch.flip(k_in, dims=(1,))
                    inputs[idx_t] = torch.flip(temp_in, dims=(1,))
                else:
                    inputs[idx_g] = g_in
                    inputs[idx_k] = k_in
                    inputs[idx_t] = temp_in
            elif input_props == "t":
                if revert_order:
                    inputs = torch.flip(temp_in, dims=(1,))
                else:
                    inputs = temp_in
            
            torch.save(inputs, prepared_dir_2ndlevel / "Inputs" / f"RUN_{new_id}.pt")

            file_label = prepared_pieces_dir / f"Label Box {box_id+1}" / f"RUN_{file_id}.pt"
            shutil.copy(file_label, prepared_dir_2ndlevel / "Labels" / f"RUN_{new_id}.pt")

def infer_single_dp(datapoint, model, settings, destination: Path):
    data = torch.load(datapoint)
    data = torch.unsqueeze(data, 0)
    y_out = model(data.to(settings.device)).to(settings.device)
    y_out = y_out.detach().cpu()
    y_out = torch.squeeze(y_out, 0)
    torch.save(y_out, destination)

def inference_levelwise(settings, datapoint_id, level:int = 1, nr_boxes: int = 10):
    if  level==1:
        output_dir = "Outputs L1"
    elif level==2:
        output_dir = f"Outputs L2 RUN_{datapoint_id}"
        
    if settings.save_inference:
        # push all datapoints through and save all outputs
        model = UNet(in_channels=len(settings.inputs)).float()
        model.load(settings.model, settings.device)
        model.eval()

        data_dir = settings.dataset_prep
        (data_dir / output_dir).mkdir(exist_ok=True)

        if level==1:
            datapoint = data_dir / "Inputs Box 0" / f"RUN_{datapoint_id}.pt"
            infer_single_dp(datapoint, model, settings, destination = data_dir / output_dir / datapoint.name)

        elif level==2:
            box_id = 1
            while box_id < nr_boxes:

                if box_id == 1:
                    datapoint = data_dir / "Outputs L1" / f"RUN_{datapoint_id}.pt"
                else:
                    datapoint = data_dir / f"Outputs L2 RUN_{datapoint_id}" / f"Box {box_id-1}.pt"
                infer_single_dp(datapoint, model, settings, destination = data_dir / output_dir / f"Box {box_id}.pt")
                
                box_id += 1
        print(f"Inference finished, outputs saved in {data_dir / output_dir}")


def reverse_norm(data, stats):
    out_max, out_min = 1, 0
    if stats["norm"] == "Rescale":
        delta = stats["max"] - stats["min"]
        return (data - out_min) / (out_max - out_min) * delta + stats["min"]
    elif stats["norm"] == "Standardize":
        return data * stats["std"] + stats["mean"]
    elif stats["norm"] is None:
        return data
    else:
        raise ValueError(f"Normalization type '{stats['Norm']}' not recognized")    
    
def pipeline_visualize(datapoint_id, settings, nr_boxes_orig:int, save:bool=False):
    ticks = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    label_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    nr_boxes_to_plot = 4

    plt.figure(figsize=(20, 5))
    concat_predict = torch.load(settings.dataset_prep / f"Outputs L1" / f"RUN_{datapoint_id}.pt")
    for box_id in range(1, nr_boxes_to_plot):
        box_predict = torch.load(settings.dataset_prep / f"Outputs L2 RUN_{datapoint_id}" / f"Box {box_id}.pt")
        concat_predict = torch.cat((concat_predict, box_predict), 1)

    # reverse norm
    info_T = yaml.safe_load(open(settings.dataset_prep / "info.yaml", "r"))["Labels"]["Temperature [C]"]
    concat_predict_normed = reverse_norm(concat_predict, info_T)
    plt.subplot(3,1,1)
    plt.imshow(concat_predict_normed.detach().cpu().numpy().T)
    plt.xticks(ticks[:nr_boxes_to_plot], label_ticks[:nr_boxes_to_plot])
    plt.title("Prediction")
    _aligned_colorbar()

    concat_orig = torch.load(settings.dataset_prep / "Label Box 0" / f"RUN_{datapoint_id}.pt")
    
    for box_id in range(1,nr_boxes_orig):
        box_orig = torch.load(settings.dataset_prep / f"Label Box {box_id}" / f"RUN_{datapoint_id}.pt")
        concat_orig = torch.cat((concat_orig, box_orig), 1)
    concat_orig_normed = reverse_norm(concat_orig, info_T)
    plt.subplot(3,1,2)
    plt.imshow(concat_orig_normed.detach().cpu().numpy().T)
    plt.xticks(ticks[:nr_boxes_to_plot], label_ticks[:nr_boxes_to_plot])
    plt.title("Original")
    _aligned_colorbar()

    plt.subplot(3,1,3)
    len_min = np.min([concat_orig_normed.shape[1], nr_boxes_to_plot*64])
    plt.imshow((concat_predict_normed[:,:len_min] - concat_orig_normed[:,:len_min]).detach().cpu().numpy().T)
    plt.xticks(ticks[:nr_boxes_to_plot], label_ticks[:nr_boxes_to_plot])
    plt.title("Difference")
    _aligned_colorbar()
    plt.tight_layout()

    if save:
        plt.savefig(settings.dataset_prep/ f"RUN_{datapoint_id}_Inference.png")
    else:
        plt.show()

def pipeline(datapoint_id: int, dataset_prep: str, inputs_2ndlevel: str, nr_boxes_orig: int, model: str):
    args = {}
    args["dataset_prep"]    = dataset_prep
    args["model"]           = "../extend_plumes by cut_pieces/model cut_4pieces separate_boxes 1st level"
    args["inputs"]          = "gksi"
    args["dataset_raw"]     = "dataset_2d_small_100dp"
    args["device"]          = "cuda:0"
    args["case"]            = "test"
    args["epochs"]          = 1
    args["destination"]     = "pipeline_1"
    args["visualize"]       = False
    args["save_inference"]  = True

    settings = SettingsTraining(**args)
    settings = prepare_data_and_paths(settings)

    inference_levelwise(settings, datapoint_id)

    args["model"]           = model
    args["inputs"]          = inputs_2ndlevel

    settings = SettingsTraining(**args)
    settings = prepare_data_and_paths(settings)

    inference_levelwise(settings, datapoint_id, level=2, nr_boxes = nr_boxes_orig)

    # visualize
    pipeline_visualize(datapoint_id, settings, nr_boxes_orig, save=True)