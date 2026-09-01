from pathlib import Path
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from utils.utils_args import load_yaml, save_yaml
from preprocessing.data_init import init_data
from preprocessing.transforms import NormalizeTransform
from processing.networks.unetVariants import UNetNoPad2, UNet
from processing.loss_fcts import SSIMLoss, LinfLoss, PATLoss
from postprocessing.metric_connectivity import connectivityLoss

def load_hyperparams(args):
    hyperparams = load_yaml(args["destination"] / "HPS_options.yaml")
    for key in hyperparams.keys():
        args[key] = hyperparams[key]["values"][0]
    return args

def init_model(args, input_channels, output_channels):
    if args["problem"] in ["allin1"]:
        if input_channels == 3 and output_channels == 1: # end-to-end, want to use same data-split
            model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"]).float()
        else:
            model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
    elif args["problem"] in ["1hp"]:
        model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], dilation=args["dilation"]).float() # older implementation
    elif args["problem"] in ["test"]:
        model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], dilation=args["dilation"]).float() # newer implementation
    model.to(args["device"])
        
    if args["case"] in ["test", "finetune"]:
        model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()
    return model

def preparation(PATH_current_data: Path, PATH_current_model: Path, PATH_destination: Path, scaling:bool=False):
    # load data and model
    args = load_hyperparams({"destination": PATH_current_model})
    args["data_prep"] = PATH_current_data
    args["model"] = PATH_current_model
    args["case"] = "test"
    args["destination"] = PATH_destination

    cla = load_yaml(PATH_current_model / "command_line_arguments.yaml")
    args["order_data"] = cla["order_data"]
    args["problem"] = cla["problem"]
    args["device"] = "cpu"
    if scaling:
        args["order_data"] = [0,0]

    # print(args)
    input_channels, output_channels, dataloaders = init_data(args)

    if "ddunet" in PATH_current_model.name.lower():
        pass
    elif any(x in PATH_current_model.name.lower() for x in ["energy", "padding", "dilated"]):
        args["problem"] = "test"
        model = init_model(args, input_channels, output_channels)
    elif any(x in PATH_current_model.name.lower() for x in ["unet",]):
        args["problem"] = "1hp"
        # model = UNet(input_channels, output_channels, **args)
        model = init_model(args, input_channels, output_channels)
    else:
        model = init_model(args, input_channels, output_channels)
    
    info = load_yaml(PATH_current_model / "info.yaml")
    norm = NormalizeTransform(info)
    # print(info)

    return dataloaders, model, norm, info, args, output_channels

def collect_metrics(PATH_current_data, PATH_current_model, PATH_destination, dataloaders, model, norm, info, args, output_channels, filename=None):
    collected_metrics = {"model": PATH_current_model.name, "data": PATH_current_data.name, "order data": args["order_data"]}
    for case, dataloader in dataloaders.items():
        collected_metrics[case] = {}
        metrics:dict = {"MSE [phys. unit^2]": MSELoss(), "MAE [phys. unit]": L1Loss(), "Linf [phys. unit]": LinfLoss(), "Huber [phys. unit]": HuberLoss(), "SSIM": SSIMLoss()}
        if output_channels == 1: # some only make sense for Temperature predictions
            metrics["MoC [--]"], metrics["PAT0.1 [%]"], metrics["PAT1.0 [%]"] = None, PATLoss(pat_thresholds=[0.1]), PATLoss(pat_thresholds=[1])
        for metric_name, metric in metrics.items():
            print(f"{filename}: Calculating {metric_name} for {case}",end=" ")
            metrics_values = []

            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(args["device"])
                targets = targets.to(args["device"])
                outputs = model(inputs).detach()
                inputs, targets = crop_to_output_size(inputs, targets, outputs)

                if metric_name in ["SSIM",]:
                    values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])
                else:
                    # unnormalize inputs and targets
                    reverse_normalization(norm, inputs, targets, outputs)
                    # calc metrics per output channel
                    if "MoC" in metric_name:
                        if targets.shape[-3] == 1: # only applicable to Temperature output
                            values = []
                            for input, output in zip(inputs, outputs):
                                dict_connectivity = connectivityLoss(input, output, id_mat_ids=info["Inputs"]["Material ID"]["index"], threshold=10.7)
                                values.append(dict_connectivity["unconnected_percentage"])
                            values = torch.mean(torch.Tensor(values))
                        else: 
                            values = torch.Tensor([0, 0]) #[torch.inf, torch.inf])
                    elif "PAT" in metric_name:
                        values = torch.mean(torch.Tensor(metric(outputs, targets).squeeze()))
                    else:
                        values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])

                metrics_values.append(values)

            assert len(dataloader) == 1, "I assumed I always have only one batch - otherwise please rethink this code"
            metrics[metric_name] = torch.mean(torch.stack(metrics_values), dim=0)# average over all batches
            print(f": average = {metrics[metric_name]}")

            collected_metrics[case][metric_name] = metrics[metric_name]
    save_yaml(collected_metrics, PATH_destination / f"{filename}.yaml")

def collect_metrics_101(path_pred, path_label, norm, info, scaling, PATH_destination, filename=None):
    collected_metrics = {"path_pred": path_pred, "path_label": path_label.name}
    try:
        outputs = torch.load(path_pred).cpu()
    except: #unet++
        outputs = torch.load(path_pred)[0]["prediction"]
        outputs = outputs.cpu()
    run = 0 if scaling else 4
    targets = torch.load(path_label / "Labels" / f"RUN_{run}.pt").permute(0,2,1).unsqueeze(0)
    inputs = torch.load(path_label / "Inputs" / f"RUN_{run}.pt").permute(0,2,1).unsqueeze(0)
    norm(outputs, data_type="Labels") 
    assert inputs.shape[-2:] == targets.shape[-2:], f"Input shape {inputs.shape} and target shape {targets.shape} do not match!"
    assert outputs.shape[-2:] == targets.shape[-2:], f"Output shape {outputs.shape} and target shape {targets.shape} do not match!"

    # unnormalize
    inputs_unnormed, targets_unnormed, outputs_unnormed = inputs.clone(), targets.clone(), outputs.clone()
    reverse_normalization(norm, inputs_unnormed, targets_unnormed, outputs_unnormed)
    
    metrics:dict = {"MSE [phys. unit^2]": MSELoss(), "MAE [phys. unit]": L1Loss(), "Linf [phys. unit]": LinfLoss(), "Huber [phys. unit]": HuberLoss(), "SSIM": SSIMLoss(),
    "MoC [--]": None, "PAT0.1 [%]": PATLoss(pat_thresholds=[0.1]), "PAT1.0 [%]": PATLoss(pat_thresholds=[1]), # these only if predictT
    }
    collected_metrics["test"] = {}
    for metric_name, metric in metrics.items():
        metrics_values = []
        print(f"{filename}: Calculating {metric_name} for test",end=" ")
        if metric_name in ["SSIM",]:
            metrics_values = torch.Tensor([metric(outputs[:,i], targets[:,i]) for i in range(outputs.shape[1])])
        elif "MoC" in metric_name:
                dict_connectivity = connectivityLoss(inputs_unnormed.squeeze(0), outputs_unnormed.squeeze(0), id_mat_ids=info["Inputs"]["Material ID"]["index"], threshold=10.7)
                metrics_values.append(dict_connectivity["unconnected_percentage"])
        elif "PAT" in metric_name:
            metrics_values = metric(outputs_unnormed.squeeze(0), targets_unnormed.squeeze(0)).squeeze()
        else:
            for channel in range(targets.shape[1]):
                metrics_values.append(metric(outputs_unnormed[:,channel,:,:], targets_unnormed[:,channel,:,:]))
        metrics[metric_name] = metrics_values
        print(f": average = {metrics[metric_name]}")

        collected_metrics["test"][metric_name] = metrics[metric_name]
    save_yaml(collected_metrics, PATH_destination / f"{filename}.yaml")


def reverse_normalization(norm, inputs, labels, outputs):
    for tmp_in in inputs:
        norm.reverse(tmp_in, data_type="Inputs")
    for tmp_tar in labels:
        norm.reverse(tmp_tar, data_type="Labels")
    for tmp_out in outputs:
        norm.reverse(tmp_out, data_type="Labels")

def crop_to_output_size(inputs=None, labels=None, outputs=None):
    required_size = outputs.shape[-2:]
    start_pos_in = ((inputs.shape[-2] - required_size[0])//2, (inputs.shape[-1] - required_size[1])//2)
    start_pos_lab = ((labels.shape[-2] - required_size[0])//2, (labels.shape[-1] - required_size[1])//2)
    if len(labels.shape) == 4:
        inputs = inputs[:,:,start_pos_in[0]:start_pos_in[0]+required_size[0], start_pos_in[1]:start_pos_in[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,:,start_pos_lab[0]:start_pos_lab[0]+required_size[0], start_pos_lab[1]:start_pos_lab[1]+required_size[1]] if labels is not None else None
    elif len(labels.shape) == 3:
        inputs = inputs[:,start_pos_in[0]:start_pos_in[0]+required_size[0], start_pos_in[1]:start_pos_in[1]+required_size[1]] if inputs is not None else None
        labels = labels[:,start_pos_lab[0]:start_pos_lab[0]+required_size[0], start_pos_lab[1]:start_pos_lab[1]+required_size[1]] if labels is not None else None
    return inputs,labels

def metrics_of_one_model(file_name, n_outputs=2, directory=Path("."), run_ids=[1,2,3,4,5]):
    if "real" in file_name(1):
        metrics = {"train": {}, "val": {}}
        means = {"train": {}, "val": {}}
        stds = {"train": {}, "val": {}}
        mins_maxs = {"train": {}, "val": {}}
    else:
        metrics = {"test": {}, "train": {}, "val": {}}
        means = {"test": {}, "train": {}, "val": {}}
        stds = {"test": {}, "train": {}, "val": {}}
        mins_maxs = {"test": {}, "train": {}, "val": {}}
    inner_metrics = {
      "model": [],
      "Huber": [[] for _ in range(n_outputs)],
      "Linf": [[] for _ in range(n_outputs)],
      "MAE": [[] for _ in range(n_outputs)],
      "MSE": [[] for _ in range(n_outputs)],
      "SSIM": [[] for _ in range(n_outputs)],
    }
    for case in metrics.keys():
        try:
            metrics[case] = deepcopy(inner_metrics)
            if n_outputs == 1:
                metrics[case]["PAT0.1"] = [[]]
                metrics[case]["PAT1.0"] = [[]]

            # load the metrics from the yaml files
            for run in run_ids:
                try:
                    file = directory / file_name(run)
                    tmp = load_yaml(file)
                    metrics[case]["model"].append(tmp["model"])
                    for name, values in tmp[case].items():
                        name = name.split(" ")[0]
                        if n_outputs == 1:
                            metrics[case][name][0].append(values)
                        if n_outputs == 2:
                            metrics[case][name][0].append(values[0])
                            metrics[case][name][1].append(values[1])
                except FileNotFoundError:
                    print(f"File {file} not found")
                    continue
            # print(metrics_test["Huber"])
            
            # calc mean and std, max and min for each metric in metrics_test for each of the 2 entries
            means_tmp = {}
            stds_tmp = {}
            mins_maxs_tmp = {}
            for name, values in metrics[case].items():
                if name == "model":
                    continue
                means_tmp[name] = []
                stds_tmp[name] = []
                mins_maxs_tmp[name] = [[] for _ in range(n_outputs)]
                for prop in range(n_outputs):
                    means_tmp[name].append(np.mean(values[prop]))
                    stds_tmp[name].append(np.std(values[prop], ddof=1))
                    mins_maxs_tmp[name][prop].append(np.min(values[prop]))
                    mins_maxs_tmp[name][prop].append(np.max(values[prop]))
                    print(f"{case} {prop} {name}: {means_tmp[name][prop]:.4f} ± {stds_tmp[name][prop]:.4f} in [{mins_maxs_tmp[name][prop][0]:.4f}, {mins_maxs_tmp[name][prop][1]:.4f}]")
            means[case] = means_tmp
            stds[case] = stds_tmp
            mins_maxs[case] = mins_maxs_tmp
        except:
            print(f"{case} does not exist for {file_name}")
            continue
    return metrics, {"means": means, "stds": stds, "mins_maxs": mins_maxs}

def plot_metrics(metrics, title):
    n_cases = len(metrics.keys())
    n_outputs = len(metrics["train"]["MAE"])
    if n_outputs == 1:
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 6))
    elif n_outputs == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        ax0 = axs[0]
    ax0.boxplot(metrics["train"]["MAE"][0], positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='blue'))
    ax0.boxplot(metrics["val"]["MAE"][0], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='green'))
    if n_cases == 3:
        ax0.boxplot(metrics["test"]["MAE"][0], positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='coral'))
    ax0.boxplot(np.sqrt(metrics["train"]["MSE"][0]), positions=[n_cases+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax0.boxplot(np.sqrt(metrics["val"]["MSE"][0]), positions=[n_cases+2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    if n_cases == 3:
        ax0.boxplot(np.sqrt(metrics["test"]["MSE"][0]), positions=[n_cases+3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    if n_cases == 3:
        ax0.set_xticks([1, 2, 3, 4, 5, 6])
        ax0.set_xticklabels(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"])
    else:
        ax0.set_xticks([1, 2, 3, 4])
        ax0.set_xticklabels(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"])
    if n_outputs == 1:
        ax0.set_ylabel("T [°C]")
    else:
        ax0.set_ylabel("vx [m/y]")
        ax0.set_title("VX Metrics")
    ax0.grid(axis='y', linestyle='--', alpha=0.7)
    if n_outputs == 2:
        axs[1].boxplot(metrics["train"]["MAE"][1], positions=[1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='blue'))
        axs[1].boxplot(metrics["val"]["MAE"][1], positions=[2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='green'))
        if n_cases == 3:
            axs[1].boxplot(metrics["test"]["MAE"][1], positions=[3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='coral'))
        axs[1].boxplot(np.sqrt(metrics["train"]["MSE"][1]), positions=[n_cases+1], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axs[1].boxplot(np.sqrt(metrics["val"]["MSE"][1]), positions=[n_cases+2], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        if n_cases == 3:
            axs[1].boxplot(np.sqrt(metrics["test"]["MSE"][1]), positions=[n_cases+3], widths=0.5, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
        if n_cases == 3:
            axs[1].set_xticks([1, 2, 3, 4, 5, 6])
            axs[1].set_xticklabels(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"])
        else:
            axs[1].set_xticks([1, 2, 3, 4])
            axs[1].set_xticklabels(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"])
        axs[1].set_ylabel("vy [m/y]")
        axs[1].set_title("VY Metrics")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    if n_cases == 3:
        plt.legend(["Train MAE", "Val MAE", "Test MAE", "Train RMSE", "Val RMSE", "Test RMSE"], loc="upper left")
    else:
        plt.legend(["Train MAE", "Val MAE", "Train RMSE", "Val RMSE"], loc="upper left")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":

    # calculate metrics for one model
    PATH_PREP_DATA = None
    PATH_MODEL = None
    PATH_DESTINATION = None
    SCALING = False #or True, if it should be evaluated on the scaling test data
    
    dataloaders, model, norm, info, args, output_channels = preparation(PATH_PREP_DATA, PATH_MODEL, PATH_DESTINATION, scaling=SCALING)
    collect_metrics(PATH_PREP_DATA, PATH_MODEL, PATH_DESTINATION, dataloaders, model, norm, info, args, output_channels)

    # exemplary use on how to calculate mean, std, min and max for several runs of one set of hyperparameters
    file_name = lambda run_id: f"metrics_run{run_id}.yaml"
    metrics = metrics_of_one_model(file_name, n_outputs=1, directory=Path("runs") / "modelA_several_runs", run_ids=[1,2,3,4,5])
    plot_metrics(metrics, "Title")