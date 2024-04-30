import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from preprocessing.domain_classes.domain import get_box_corners
from torch.nn import MSELoss, modules

from postprocessing.visualization import _aligned_colorbar

matplotlib.use('TkAgg')


def load_data_1hpnn(folder):
    nns = {}
    nn_path = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/before_2023_10_domain/"
    for id in range(1, 13):
        nns[id] = {"label": torch.load(nn_path + folder + "/RUN_" + str(id) + "_label.pt")[0],
                "pred": torch.load(nn_path + folder + "/RUN_" + str(id) + "_prediction.pt"),
                "inputs": torch.load(nn_path + folder + "/RUN_" + str(id) + "_inputs.pt"),
                }
    return nns

def set_paths_2hpnn():
    data_path = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_NN/"
    destination_path = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/before_2023_10_domain/"
    return data_path, destination_path

def load_data_2hpnn(dataset_name, destination_path):
    nns_2hp = {}
    for file in os.listdir(destination_path+dataset_name):
        if file.endswith("_label.pt"):
            id = str(int(file.split("_")[1]))+str(int(file.split("_")[3]))
            name = file.replace("_label.pt", "")
            try:
                pred = torch.Tensor(torch.load(destination_path+dataset_name + "/" + name + "_prediction.pt")[0,0])
            except:
                pred = torch.Tensor(torch.load(destination_path+dataset_name + "/" + name + "_prediction.pt")[0])
            label = torch.Tensor(torch.load(destination_path+dataset_name + "/" + name + "_label.pt")[0])
            nns_2hp[id] = {"label": label, "pred": pred}   
    return nns_2hp

def affected_area(input_field: np.ndarray, threshold: float = None) -> np.ndarray:
    return input_field >= threshold

def get_hps_and_box_mask(field_shape:np.ndarray, material_ids:np.ndarray):
    distance_hp_corner = [7, 23]
    size_hp_box = [16, 256]
    box_mask = np.zeros(field_shape)
    pos_hps = np.array(np.where(material_ids == np.max(material_ids))).T
    pos_hps += 1
    hps = []
    for idx in range(len(pos_hps)):
        pos_hp = pos_hps[idx]
        corner_ll = get_box_corners(pos_hp, size_hp_box, distance_hp_corner, field_shape)
        box_mask[corner_ll[0]-1 : size_hp_box[0]-1, corner_ll[1]-1 : size_hp_box[1]-1] = 1
        hps.append({"pos" : pos_hps[idx], "mask": box_mask.copy()})
        box_mask = np.zeros(field_shape)
    return hps

def eval_affected_area_1hpnn(nns, plot_bool:bool = False):
    loss_func: modules.loss._Loss = MSELoss()
    sum_affected_cells: dict = {}
    calc_loss_measures: bool = True
    mse_closs = 0.0
    mae_closs = 0.0
    max_temp_diff = 0.0
    for threshold in [10.7, 11.6, 14.6]: 
        counter = 0
        sum_affected_cells[threshold] = 0
        for id, datapoint in nns.items():
            label = datapoint["label"].T
            pred = datapoint["pred"].T
            mat_ids = datapoint["inputs"][3].T
            hps = get_hps_and_box_mask(label.shape, mat_ids)
            for hp in hps:
                # currently just the inside
                i, j = np.where(hp["mask"])
                indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                        np.arange(min(j), max(j) + 1),
                        indexing='ij')
                hp["label"] = label[indices]
                hp["pred"] = pred[indices]
                area = affected_area(hp["label"], threshold)
                area_pred = affected_area(hp["pred"], threshold)
                area_diff = area ^ area_pred
                sum_affected_cells[threshold] += np.sum(area_diff)
                if calc_loss_measures:
                    mse_closs += loss_func(torch.Tensor(hp["pred"]), torch.Tensor(hp["label"]))
                    mae_closs += torch.mean(torch.abs(torch.Tensor(hp["pred"]) - torch.Tensor(hp["label"])))
                    max_temp_diff += abs(np.max(hp["pred"]) - np.max(hp["label"]))/np.max(hp["label"])

                counter += 1

                if plot_bool:
                    plt.subplot(5, 1, 1)
                    plt.imshow(hp["label"])
                    _aligned_colorbar()
                    plt.title("label")
                    plt.subplot(5, 1, 2)
                    plt.imshow(hp["pred"])
                    _aligned_colorbar()
                    plt.title("pred")
                    plt.subplot(5, 1, 3)
                    plt.imshow(area)
                    _aligned_colorbar()
                    plt.title("label")
                    plt.subplot(5, 1, 4)
                    plt.imshow(area_pred)
                    _aligned_colorbar()
                    plt.title("pred")
                    plt.subplot(5, 1, 5)
                    plt.imshow(area_diff)
                    _aligned_colorbar()
                    plt.show()
                    # plt.savefig(f"pics_affected_area/1hpnn/test{counter}_{str(threshold).replace('.', '_')}.pgf", format="pgf")
                    plt.close()
            
        sum_affected_cells[threshold] /= counter
        if calc_loss_measures:
            mse_closs /= counter
            mae_closs /= counter
            max_temp_diff /= counter
        calc_loss_measures = False
    return sum_affected_cells, mse_closs, mae_closs, max_temp_diff


def eval_affected_area_2hpnn(nns, plot_bool:bool = False):
    loss_func: modules.loss._Loss = MSELoss()
    sum_affected_cells: dict = {}
    calc_rmse: bool = True
    for threshold in [10.7, 11.6, 14.6]: 
        counter = 0
        sum_affected_cells[threshold] = 0
        if calc_rmse:
            mse_closs = 0.0
        for id, datapoint in nns.items():
            label = datapoint["label"].T
            pred = datapoint["pred"].T.detach().to("cpu")
            area = affected_area(label, threshold)
            area_pred = affected_area(pred, threshold)
            area_diff = area ^ area_pred
            
            sum_affected_cells[threshold] += torch.sum(area_diff)
            if calc_rmse:
                mse_closs += loss_func(torch.Tensor(pred), torch.Tensor(label))
            counter += 1

            if plot_bool:
                plt.subplot(5, 1, 1)
                plt.imshow(label)
                _aligned_colorbar()
                plt.title("label")
                plt.subplot(5, 1, 2)
                plt.imshow(pred)
                _aligned_colorbar()
                plt.title("pred")
                plt.subplot(5, 1, 3)
                plt.imshow(area)
                _aligned_colorbar()
                plt.title("label")
                plt.subplot(5, 1, 4)
                plt.imshow(area_pred)
                _aligned_colorbar()
                plt.title("pred")
                plt.subplot(5, 1, 5)
                plt.imshow(area_diff)
                _aligned_colorbar()
                plt.show()
                # plt.savefig(f"pics_affected_area/2hpnn/test{id}_{str(threshold).replace('.', '_')}.pgf", format="pgf")
                plt.close()
        sum_affected_cells[threshold] = sum_affected_cells[threshold]/counter
        if calc_rmse:
            mse_closs /= counter
        calc_rmse = False
    return sum_affected_cells, mse_closs

def eval_max_temp(nns):
    overall_max_temp = 0
    for id, datapoint in nns.items():
        try:
            max_pred = torch.max(datapoint["pred"])
            max_label = torch.max(datapoint["label"])
        except:
            max_pred = np.max(datapoint["pred"])
            max_label = np.max(datapoint["label"])
        max_temp_diff = abs(max_pred - max_label)/max_label
        overall_max_temp += max_temp_diff
    overall_max_temp /= len(nns)
    return overall_max_temp

if __name__ == "__main__":
    # 1HPNN
    folder = "BENCHMARK_DOMAIN2_gksi_1HPNN"
    nns_1hp = load_data_1hpnn(folder)
    sums, mse_closs, mae_closs, max_temp_diff = eval_affected_area_1hpnn(nns_1hp, plot_bool=False)
    sums_string = ""
    for value in sums.values():
        sums_string += f"{np.round(value,2)}~cells & "
    print(f"{np.sqrt(mse_closs):.4f}~°C  & {mse_closs:.4f}~°C & {(max_temp_diff*100):.3f}~\% & {sums_string[:-2]}")
    #results: 0.7137~°C & 0.688~\% & 189.38~cells & 322.71~cells & 215.12~cells 

    ## 2HPNN
    dataset_name = "BENCHMARK_DOMAIN2_gksi_2HPNN_boxes"

    print(f"model: {0},\ndataset: {dataset_name}")
    data_path, destination_path = set_paths_2hpnn()
    nns_2hp = load_data_2hpnn(dataset_name, destination_path)
    sums, mse_closs = eval_affected_area_2hpnn(nns_2hp, plot_bool=True)
    max_temp_diff = eval_max_temp(nns_2hp)
    sums_string = ""
    for value in sums.values():
        sums_string += f"{np.round(value,2)}~cells & "
    print(f"{np.sqrt(mse_closs):.4f}~°C & {(max_temp_diff*100):.3f}~\% & {sums_string[:-2]}")
    #results: 0.1120~°C & 0.866~\% & 23.829999923706055~cells & 25.290000915527344~cells & 32.33000183105469~cells 