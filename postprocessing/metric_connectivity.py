import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from preprocessing.datasets.dataset import DataPoint
from processing.networks.unetVariants import UNetNoPad2
from processing.networks.unet import UNet
from postprocessing.visualization import reverse_norm_one_dp
from postprocessing.visu_utils import _aligned_colorbar
from postprocessing.cmap_jp import new_cmap

def unify_size(inputs, required_size):
    # expect square
    start = int((len(inputs[0,0]) - required_size)//2)
    inputs = inputs[:,start:start+required_size, start:start+required_size]
    return inputs

def masking(data, threshold:float = 10.7):
    mask = data > threshold
    masked_data = data[mask]
    return mask, masked_data

# flood fill
def in_bounds(curr_pos, size):
    tmp = (curr_pos >= 0) & (curr_pos < size)
    return tmp[0] & tmp[1]
 
def select_in_bounds(curr_pos, size):
    return curr_pos.T[in_bounds(curr_pos, size)].T
 
def filter_visited(curr_pos, visited, field):
    return curr_pos.T[(~visited[curr_pos[0],curr_pos[1]] & field[curr_pos[0],curr_pos[1]])].T
 
def step(active_indices, visited, field):
    size = visited.shape[0]
    visited[active_indices[0],active_indices[1]] = True
    up = active_indices.copy()
    up[1] += 1
    down = active_indices.copy()
    down[1] -= 1
    right = active_indices.copy()
    right[0] += 1
    left = active_indices.copy()
    left[0] -= 1
    up = select_in_bounds(up, size)
    down = select_in_bounds(down, size)
    left = select_in_bounds(left, size)
    right = select_in_bounds(right, size)
    up = filter_visited(up, visited, field)
    visited[up[0],up[1]] = True
    down = filter_visited(down, visited, field)
    visited[down[0],down[1]] = True
    left = filter_visited(left, visited, field)
    visited[left[0],left[1]] = True
    right = filter_visited(right, visited, field)
    visited[right[0],right[1]] = True
 
    new_indices = np.concatenate((up, down, left, right), axis=1)
    return new_indices
 
def flood_fill(active_indices, field):
    visited=np.zeros_like(field,dtype=bool)
    i = 0
    while len(active_indices[0]) > 0:
        i += 1
        active_indices = step(active_indices, visited, field)
    return visited

def test_flood_fill():
    np.random.seed(0)
    size = 2560
    n_start = 100
    field_to_fill = np.ones((size, size), dtype=bool)
    active_indices = np.random.randint(0, size - 1, (2, n_start))
    flood_fill(active_indices, field_to_fill)

def connectivity_field_flood(mat_ids_unnormed, mask_output):
    hps = np.where(mat_ids_unnormed == 2)
    hps = np.array(hps)
    connectivity_field  = flood_fill(hps, mask_output[0].cpu().numpy())

    unconnected_cells = connectivity_field ^ np.array(mask_output[0])
    connected_cells = np.sum(np.array(mask_output[0]))
    return connectivity_field, unconnected_cells, connected_cells

def calc_connectivity(model_path:Path, data_path:Path, data_id: int, model:UNet, id_mat_ids:int, threshold:float):
    # Data and model loading
    model.load(model_path)
    data = DataPoint(data_path, idx=data_id)
    inputs, label = data[0]
    output = model.infer(inputs.unsqueeze(0))

    # Data preparation
    inputs, data, output = reverse_norm_one_dp(inputs, label, output, data.norm)
    required_size = len(output[0,0])
    inputs = unify_size(inputs, required_size)
    label = unify_size(label, required_size)
    output = unify_size(output, required_size)

    # Data masking at threshold
    mask_label, masked_label = masking(label, threshold)
    mask_output, masked_output = masking(output, threshold)

    connectivity_label, unconn_cells_label, conn_cells_label = connectivity_field_flood(inputs[id_mat_ids], mask_label)
    ratio_label = np.sum(unconn_cells_label)/conn_cells_label
    # print("Ratio: ", ratio_label, ", unconnected cells: ", np.sum(unconn_cells_label), ", connected cells: ", conn_cells_label)
    connectivity_output, unconn_cells_output, conn_cells_output = connectivity_field_flood(inputs[id_mat_ids], mask_output)
    ratio_output = np.sum(unconn_cells_output)/conn_cells_output
    # print("Ratio: ", np.sum(unconn_cells_output)/conn_cells_output, ", unconnected cells: ", np.sum(unconn_cells_output), ", connected cells: ", conn_cells_output)

    return {"field" : label[0],
                      "connectivity" : connectivity_label,
                      "unconnected_cells" : unconn_cells_label,
                      "ratio" : ratio_label}, {"field" : output[0],
                      "connectivity" : connectivity_output,
                      "unconnected_cells" : unconn_cells_output,
                      "ratio" : ratio_output}

if __name__ == "__main__":
    model_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/paper24 finals/naive_approach_unetnopad")
    data_path = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_giant_100hp_varyK inputs_pki outputs_t")
    run_id = 2
    id_mat_ids = 2
    model = UNetNoPad2(in_channels=3, out_channels=1, depth=3, init_features=32, kernel_size=5).float()

    threshold = 10.7
    dict_connectivity = calc_connectivity(model_path, data_path, model, id_mat_ids, threshold)