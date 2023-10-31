import matplotlib.pyplot as plt
import torch
import pathlib
import numpy as np
import torch
import yaml

from utils.prepare_paths import Paths2HP, set_paths_2hpnn
from utils.visualization import _aligned_colorbar
from preprocessing.prepare_2ndstage import prepare_dataset_for_2nd_stage
from domain_classes.domain import get_box_corners

inputs = torch.load("/home/pelzerja/Development/datasets_prepared/1hp_domain/boxes_dummy_inputs.pt")
dummy_box_sdf = inputs[2]
box_size = dummy_box_sdf.shape
pos_hp_box_relative = np.array(np.where(inputs[3]==1)).squeeze()
distance = np.array(np.where(dummy_box_sdf == 0)).squeeze() - pos_hp_box_relative
distance = np.sqrt(distance[0]**2 + distance[1]**2)

domain_path = pathlib.Path("/home/pelzerja/Development/datasets_prepared/1hp_domain/domain_1hp_1dp inputs_gksi/")
inputs_domain = torch.load(domain_path / "Inputs/RUN_0.pt")
domain_shape = inputs_domain[0].shape

with open(domain_path / "info.yaml", "r") as file:
    infos_domain = yaml.load(file)
idx_mat_id = infos_domain["Inputs"]["Material ID"]["index"]
idx_sdf = infos_domain["Inputs"]["SDF"]["index"]

pos_hp_domain_absolute = np.array(np.where(inputs_domain[idx_mat_id]==1)).squeeze()
domain_sdf = np.zeros(domain_shape)

for x in range(domain_sdf.shape[0]):
    for y in range(domain_sdf.shape[1]):
        if np.sqrt(x**2+y**2) <= distance:
            domain_sdf[x,y] = torch.linalg.norm(torch.tensor([x,y]).float() - pos_hp_domain_absolute)


for x in range(domain_sdf.shape[0]):
    for y in range(domain_sdf.shape[1]):
        if np.sqrt(x**2+y**2) <= distance:
            domain_sdf[x,y] = 1 - domain_sdf[x,y] / domain_sdf.max()

inputs_domain[idx_sdf] = torch.tensor(domain_sdf)

domain_path = pathlib.Path("/home/pelzerja/Development/datasets_prepared/1hp_domain/domain_1hp_1dp inputs_gksi sdf_modified/")
torch.save(inputs_domain, domain_path / "Inputs/RUN_0.pt")

if False:
    plt.imshow(inputs_domain[idx_sdf].T)
    _aligned_colorbar()
    plt.show()