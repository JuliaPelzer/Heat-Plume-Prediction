import argparse
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml

from postprocessing.visu_utils import _aligned_colorbar


def prepare_domain_sdf(orig_domain_name: str = "domain_1hp_1dp inputs_gksi"):
    inputs = torch.load("/home/pelzerja/pelzerja/test_nn/datasets_prepared/1hp_domain/boxes_dummy_inputs.pt")
    dummy_box_sdf = inputs[2]
    box_size = dummy_box_sdf.shape
    pos_hp_box_relative = np.array(np.where(inputs[3]==1)).squeeze()
    distance = np.array(np.where(dummy_box_sdf == 0)).squeeze() - pos_hp_box_relative
    distance = np.sqrt(distance[0]**2 + distance[1]**2)

    domain_path = pathlib.Path("/home/pelzerja/pelzerja/test_nn/datasets_prepared/1hp_domain") / orig_domain_name

    destination_name = f"{orig_domain_name} sdf_modified"
    destination_path = domain_path.parent / destination_name
    try:
        shutil.copytree(domain_path, destination_path)
    except FileExistsError:
        print(f"{destination_path} already exists")

    with open(domain_path / "info.yaml", "r") as file:
        infos_domain = yaml.load(file, Loader=yaml.FullLoader)
    idx_mat_id = infos_domain["Inputs"]["Material ID"]["index"]
    idx_sdf = infos_domain["Inputs"]["SDF"]["index"]


    runs = (domain_path/"Inputs").iterdir()
    for path in tqdm.tqdm(runs):
        if path.is_file():
            inputs_domain = torch.load(path)
        domain_shape = inputs_domain[0].shape

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
        torch.save(inputs_domain, destination_path / "Inputs" / path.name)

        if False:
            plt.imshow(inputs_domain[idx_sdf].T)
            _aligned_colorbar()
            plt.show()

def domain_limited_sdf(orig_domain_name: str = "domain_1hp_1dp inputs_gksi"):

    inputs = torch.load("/home/pelzerja/pelzerja/test_nn/datasets_prepared/1hp_domain/boxes_dummy_inputs.pt")
    dummy_box_sdf = inputs[2]
    box_size = dummy_box_sdf.shape
    pos_hp_box_relative = np.array(np.where(inputs[3]==1)).squeeze()
    distances = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    distances["left"] = pos_hp_box_relative[0]
    distances["right"] = box_size[0] - pos_hp_box_relative[0]
    distances["top"] = pos_hp_box_relative[1]
    distances["bottom"] = box_size[1] - pos_hp_box_relative[1]
    domain_path = pathlib.Path("/home/pelzerja/pelzerja/test_nn/datasets_prepared/1hp_domain") / orig_domain_name

    destination_name = f"{orig_domain_name} sdf_modified_v2"
    destination_path = domain_path.parent / destination_name
    try:
        shutil.copytree(domain_path, destination_path)
    except FileExistsError:
        print(f"{destination_path} already exists")

    with open(domain_path / "info.yaml", "r") as file:
        infos_domain = yaml.load(file, Loader=yaml.FullLoader)
    idx_mat_id = infos_domain["Inputs"]["Material ID"]["index"]
    idx_sdf = infos_domain["Inputs"]["SDF"]["index"]


    runs = (domain_path/"Inputs").iterdir()
    for path in tqdm.tqdm(runs):
        if path.is_file():
            inputs_domain = torch.load(path)
        domain_shape = inputs_domain[0].shape

        pos_hp_domain_absolute = np.array(np.where(inputs_domain[idx_mat_id]==1)).squeeze()
        domain_sdf = np.zeros(domain_shape)
        for x in range(box_size[0]):
            for y in range(box_size[1]):
                domain_sdf[x+pos_hp_domain_absolute[0]-distances["left"], y+pos_hp_domain_absolute[1]-distances["top"]] = dummy_box_sdf[x,y]

        # domain_sdf2 = np.zeros(domain_shape)
        # for x in range(domain_sdf2.shape[0]):
        #     for y in range(domain_sdf2.shape[1]):
        #         if x >= pos_hp_domain_absolute[0] - distances["left"] and x < pos_hp_domain_absolute[0] + distances["right"] and y >= pos_hp_domain_absolute[1] - distances["top"] and y < pos_hp_domain_absolute[1] + distances["bottom"]:
        #             domain_sdf2[x,y] = torch.linalg.norm(torch.tensor([x,y]).float() - pos_hp_domain_absolute)
        #             # domain_sdf2[x,y] = torch.linalg.norm(torch.tensor([x,y]).float() - pos_hp_domain_absolute)

        # for x in range(domain_sdf2.shape[0]):
        #     for y in range(domain_sdf2.shape[1]):
        #         if x >= pos_hp_domain_absolute[0] - distances["left"] and x < pos_hp_domain_absolute[0] + distances["right"] and y >= pos_hp_domain_absolute[1] - distances["top"] and y < pos_hp_domain_absolute[1] + distances["bottom"]:
        #             domain_sdf2[x,y] = 1 - domain_sdf2[x,y] / domain_sdf2.max()


        inputs_domain[idx_sdf] = torch.tensor(domain_sdf)
        torch.save(inputs_domain, destination_path / "Inputs" / path.name)

        # if True:
        #     plt.subplot(1,3,1)
        #     plt.imshow(inputs_domain[idx_sdf].T)
        #     _aligned_colorbar()
        #     plt.subplot(1,3,2)
        #     plt.imshow(domain_sdf2.T)
        #     _aligned_colorbar()
        #     plt.subplot(1,3,3)
        #     plt.imshow(inputs_domain[idx_sdf].T - domain_sdf2.T)
        #     _aligned_colorbar()
        #     plt.show()

        # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="domain_1hp_1dp inputs_gksi")
    args = parser.parse_args()

    domain_limited_sdf(args.domain)