# %%
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from datetime import datetime
import pathlib
import matplotlib.pyplot as plt

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import load_yaml, save_yaml, make_paths
from postprocessing.visu_utils import _aligned_colorbar
import preprocessing.preprocessing as prep
from main import read_cla
from processing.networks.unet import UNet


# %%
%reload_ext autoreload
%autoreload 2

# %% [markdown]
# ## functions for data loading + preparing

# %%
## prepare data for inputs for streamlines: ik_Txy
def prepare_ik_Txy(dataset_name:str, problem:str="allin1"):
    args = read_cla(pathlib.Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/dummy_prep_ik_Txy"))
    args["data_raw"] = f"/scratch/sgs/pelzerja/datasets/{problem}/{dataset_name}"
    args["data_prep"] = f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy"

    make_paths(args) # and check if data / model exists
    prep.preprocessing(args) # and save info.yaml in model folder

## prepare inputs for streamlines: ixyk_T
def load_data(dataset_name, runid, problem:str="allin1"):
    info = load_yaml(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/info.yaml")
    norm = NormalizeTransform(info)

    inputs = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/Inputs/RUN_{runid}.pt")
    inputs_normed = deepcopy(inputs)
    norm.reverse(inputs, "Inputs")

    labels = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/Labels/RUN_{runid}.pt")
    labels_normed = deepcopy(labels)
    norm.reverse(labels, "Labels")

    args = load_yaml(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/args.yaml")
    print("current:", labels.shape, inputs_normed.shape)

    return inputs, inputs_normed, labels, labels_normed, info, args

def build_new_inputs_and_outputs(inputs_normed, labels_normed, indices):
    dummy_field = torch.zeros_like(inputs_normed[0])

    inputs_new = torch.cat([inputs_normed[indices["mat_id"]].unsqueeze(0), labels_normed[indices["vx"]].unsqueeze(0), labels_normed[indices["vy"]].unsqueeze(0), dummy_field.unsqueeze(0), dummy_field.unsqueeze(0), inputs_normed[indices["k"]].unsqueeze(0)], dim=0)
    
    labels_new = labels_normed[indices["T"]].unsqueeze(0)
    print("new:", inputs_new.shape, labels_new.shape)
    # change inputs_new dtype to float32
    inputs_new = inputs_new.float()
    return inputs_new, labels_new

def build_new_args(args):
    args["inputs"] = [
    "Material ID",
    "Liquid X-Velocity [m_per_y]",
    "Liquid Y-Velocity [m_per_y]",
    "Streamlines",
    "Streamlines Fade",
    "Permeability X [m^2]",
    ]

def build_new_info(info):
    info["Inputs"]["Streamlines"] = {
    "index": 3,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Streamlines Fade"] = {
    "index": 4,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Permeability X [m^2]"]["index"] = 5
    info["Inputs"]["Liquid X-Velocity [m_per_y]"] = info["Labels"]["Liquid X-Velocity [m_per_y]"]
    info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"] = 1
    info["Inputs"]["Liquid Y-Velocity [m_per_y]"] = info["Labels"]["Liquid Y-Velocity [m_per_y]"]
    info["Inputs"]["Liquid Y-Velocity [m_per_y]"]["index"] = 2

    info["Labels"] = info["Labels"] = {"Temperature [C]": info["Labels"]["Temperature [C]"]}

def extract_mat_ids_and_velos(inputs, label, indices):
    mat_ids = inputs[indices["mat_id"]].numpy()
    vx_real = label[indices["vx"]].numpy()
    vy_real = label[indices["vy"]].numpy()
    return mat_ids, vx_real, vy_real

def extract_required_data(dataset_name, runid, problem):
    indices = {
        "mat_id": 0,# input
        "k": 1,     # input
        "T": 0,     # label
        "vx": 1,    # label
        "vy": 2,    # label
    }

    inputs, inputs_normed, label, label_normed, info, args = load_data(dataset_name, runid, problem)
    mat_ids, vx_real, vy_real = extract_mat_ids_and_velos(inputs, label, indices)
    
    dims = mat_ids.shape

    return inputs_normed, label_normed, args, info, mat_ids, vx_real, vy_real, dims, indices


# %% [markdown]
# ## data processing + streamline calculation

# %%
def calc_streamline(x, y, vx, vy, start=[5,14], t_end=27.5, t_steps=1000):
    fx = RegularGridInterpolator((x,y), vx, bounds_error=False, fill_value=None, method="linear")
    fy = RegularGridInterpolator((x,y), vy, bounds_error=False, fill_value=None, method="linear")

    # define the velocity function to be integrated:
    def f(t, y):
        return np.squeeze([fy(y), fx(y)])

    # Solve for start point
    sol = solve_ivp(f, [0, t_end], start, t_eval=np.linspace(0,t_end,t_steps), method="Radau")

    sol_x = sol.y[0]
    sol_y = sol.y[1]
    
    # cut sol_y, sol_x if extend x.max(),y.max()
    sol_x = sol_x[sol_x <= x.max()]
    sol_y = sol_y[sol_y <= y.max()]

    length = np.min([sol_x.shape[0], sol_y.shape[0]])
    sol_x = sol_x[:length]
    sol_y = sol_y[:length]
    t = sol.t[:length]

    return sol_x, sol_y, t

# %%
# def draw_streamlines(image_data:np.array, streamlines:list, faded:bool=False):
#     time = datetime.now()
#     fig = plt.figure(frameon=False, figsize=(25.6,25.6))
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(image_data, aspect='auto', cmap="gray")
#     for streamline in streamlines:
#         if faded:
#             c=streamline[2][::-1]
#         else:
#             c="white"
#         plt.scatter(streamline[1], streamline[0], s=1, c=c, marker="s")
    
#     fig.canvas.draw()
#     # save to numpy array
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     # rgb data to gray scale
#     data = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])
#     data = data / data.max()
#     plt.close()
#     print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
#     return data

def draw_streamlines2(image_data:np.array, streamlines:list, faded:bool=False):
    time = datetime.now()
    for streamline in streamlines:
        if faded:
            val = streamline[2][::-1]
            val = val / val.max()
        else:
            val = 1
        image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] = val
    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return image_data

def make_streamlines(mat_ids, vx_real, vy_real, dims):
    pos_hps = np.array(np.where(mat_ids == 2)).T.astype(float)
    print("Number of heat pumps: ", pos_hps.shape[0])
    pos_hps += np.array([0.5,0.5]) # cell-center offset
    x,y = (np.arange(0,dims[0]),np.arange(0,dims[1]))

    time = datetime.now()
    streamlines = []
    for hp in tqdm(pos_hps, desc="Calculating streamlines"):
        sol = calc_streamline(x, y, vx_real/5, vy_real/5, hp, t_end=27.5, t_steps=10_000)
        streamlines.append(sol)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")

    streamlines_ohe = draw_streamlines2(np.zeros(dims), streamlines, faded=False)
    # print(streamlines_ohe.shape)

    streamlines_faded = draw_streamlines2(np.zeros(dims), streamlines, faded=True)
    # print(streamlines_faded.shape)

    return torch.tensor(streamlines_ohe), torch.tensor(streamlines_faded)

# %%
## storing data
def save_new_datapoint(destination, runid, inputs_new:torch.Tensor, labels_new:torch.Tensor=None):
    (destination/"Inputs").mkdir(exist_ok=True, parents=True)
    (destination/"Labels").mkdir(exist_ok=True, parents=True)
    torch.save(inputs_new, destination / "Inputs"/ f"RUN_{runid}.pt")
    if labels_new != None:
        torch.save(labels_new, destination / "Labels" / f"RUN_{runid}.pt")

def save_args_info(destination, args, info):
    build_new_info(info)
    print(info)
    save_yaml(info, destination / "info.yaml")

    build_new_args(args)
    print(args)
    save_yaml(args, destination / "args.yaml")

# %% [markdown]
# ## call the functions

# %%
# ## run data loading+preparing
# # dataset_name = "benchmark_dataset_allin1_large"
# dataset_name = "dataset_giant_100hp_varyK"
# problem = "allin1"

# # dataset_name = "bm_large_10dp_1hp_2"

# # TODO manually: set output_vars in prepare_dataset.py to Txy
# prepare_ik_Txy(dataset_name, problem)

# for runid in [2]: #1,2,4]:
#     # TODO manually set in prepare_dataset.py the output_variables to Txy
#     inputs_normed, label_normed, args, info, mat_ids, vx_real, vy_real, dims, indices = extract_required_data(dataset_name, runid, problem)
#     inputs_new, labels_new = build_new_inputs_and_outputs(inputs_normed, label_normed, indices)
#     streamlines_ohe, streamlines_faded = make_streamlines(mat_ids, vx_real, vy_real, dims)
#     destination_name = f"{dataset_name} inputs_ixyssk"

#     idx_s = 3
#     idx_sf = 4
#     inputs_new[idx_s] = streamlines_ohe.unsqueeze(0)
#     inputs_new[idx_sf] = streamlines_faded.unsqueeze(0)
#     print("new: ", inputs_new.shape, labels_new.shape)

#     # destination = pathlib.Path(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{destination_name}", exist_ok=True, parents=True)
#     # save_new_datapoint(destination, runid, inputs_new, labels_new)
    
# # save_args_info(destination, args, info)

# %%
# model_tmp = UNet(3,2)
# model_tmp.load(pathlib.Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/model_predict_v_pki"))
# model_tmp.eval()
# data_in_tmp = torch.load("/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_giant_100hp_varyK inputs_pki outputs_vv/Inputs/RUN_2.pt")
# pred_tmp = model_tmp(data_in_tmp.unsqueeze(0)).squeeze(0).detach().numpy()
# print(pred_tmp.shape)

# plt.figure(figsize=(8,8))
# plt.subplot(4,1,1)
# plt.imshow(vx_real[2000:,2300:2410].T, cmap="RdBu_r")
# plt.title("True x-velocity")
# plt.ylabel("x [cells]")
# # _aligned_colorbar()

# plt.subplot(4,1,2)
# plt.imshow(pred_tmp[0,2000:,2300:2410].T,cmap="RdBu_r")
# plt.title("Predicted x-velocity")
# plt.ylabel("x [cells]")
# # _aligned_colorbar()

# plt.subplot(4,1,3)
# plt.imshow(vy_real[2000:,2300:2410].T, cmap="RdBu_r")
# plt.title("True y-velocity")
# plt.ylabel("x [cells]")
# # _aligned_colorbar()

# plt.subplot(4,1,4)
# plt.imshow(pred_tmp[1,2000:,2300:2410].T,cmap="RdBu_r")
# plt.title("Predicted y-velocity")
# plt.ylabel("x [cells]")
# plt.xlabel("y [cells]")
# # _aligned_colorbar()

# plt.tight_layout()
# plt.savefig("velos_correct_ausschnitt_pki.png")
# plt.show()

# %%
plt.figure(figsize=(8,8))
plt.subplot(4,1,1)
plt.imshow(vx_real[2000:,2300:2410].T, cmap="RdBu_r")
plt.title("True x-velocity")
plt.ylabel("x [cells]")
# _aligned_colorbar()

plt.subplot(4,1,2)
plt.imshow(all_data_loaded[1,2000:,2300:2410].T.numpy(),cmap="RdBu_r")
plt.title("Predicted x-velocity")
plt.ylabel("x [cells]")
# _aligned_colorbar()

plt.subplot(4,1,3)
plt.imshow(vy_real[2000:,2300:2410].T, cmap="RdBu_r")
plt.title("True y-velocity")
plt.ylabel("x [cells]")
# _aligned_colorbar()

plt.subplot(4,1,4)
plt.imshow(all_data_loaded[2,2000:,2300:2410].T.numpy(),cmap="RdBu_r")
plt.title("Predicted y-velocity")
plt.ylabel("x [cells]")
plt.xlabel("y [cells]")
# _aligned_colorbar()

plt.tight_layout()
# plt.savefig("streamlines_offbyhalfcell_ausschnitt.png")
# plt.savefig("velos_correct_ausschnitt.png")
plt.show()

# %%
print(inputs_new.shape, labels_new.shape)
for idx, prop in enumerate(inputs_new):
    print(idx, prop.shape)
    plt.subplot(1,len(inputs_new)+1, idx+1)
    plt.imshow(prop.numpy())
    plt.colorbar()
plt.subplot(1,len(inputs_new)+1, len(inputs_new)+1)
plt.imshow(labels_new[0].numpy())
plt.colorbar()
plt.savefig("testi.png")


# %% [markdown]
# ## prepare ivvssk

# %%
from distutils.dir_util import copy_tree

from processing.networks.unet import UNet

# %%
## run data loading+preparing
dataset_name = "dataset_giant_100hp_varyK"
model_name = "predict_v/ref_model_train_vv_on1dp" #ref_model_predict_v_lr_scheduler"
destination = f"/scratch/sgs/pelzerja/datasets_prepared/allin1/{dataset_name} inputs_ivvssk"
copy_tree(f"/scratch/sgs/pelzerja/datasets_prepared/allin1/{dataset_name} inputs_ixyssk", destination)
destination = pathlib.Path(destination)

args = load_yaml(destination / "args.yaml")
args["inputs"][1] = f"Liquid X-Velocity [m_per_y] - predicted by '{model_name}'"
args["inputs"][2] = f"Liquid Y-Velocity [m_per_y] - predicted by '{model_name}'"
save_yaml(args, destination / "args.yaml")

model_path = pathlib.Path(f"/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/{model_name}")
model =  UNet(in_channels=3, out_channels=2)
model.load(model_path)
info_model = load_yaml(model_path / "info.yaml")

for runid in [2]:
    data_in_model = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_giant_100hp_varyK inputs_pki outputs_vv/Inputs/RUN_{runid}.pt")
    vv_out = model(data_in_model.unsqueeze(0)).detach().squeeze(0)
    print(vv_out.shape)
    norm_model = NormalizeTransform(info_model)
    norm_model.reverse(vv_out, "Labels")
    inputs = torch.load(destination/"Inputs"/f"RUN_{runid}.pt")
    norm = NormalizeTransform(load_yaml(destination/"info.yaml"))
    norm.reverse(inputs, "Inputs")
    inputs[1] = vv_out[0]
    inputs[2] = vv_out[1]
    print(inputs.shape)
    inputs = inputs.numpy()

    streamlines_ohe, streamlines_faded = make_streamlines(mat_ids=inputs[0], vx_real=inputs[1], vy_real=inputs[2], dims=inputs[0].shape)

    idx_s = 3
    idx_sf = 4
    # norm inputs acc. to info
    inputs_normed = norm(torch.tensor(inputs), "Inputs")
    inputs_normed[idx_s] = streamlines_ohe.unsqueeze(0)
    inputs_normed[idx_sf] = streamlines_faded.unsqueeze(0)
    # print("new: ", inputs_normed.shape, labels_new.shape)

    # save_new_datapoint(destination, runid, inputs_normed)
    # break

# %%
streamlines_faded_real = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_giant_100hp_varyK inputs_ixyssk/Inputs/RUN_{runid}.pt")[4]
print(streamlines_faded_real.shape)

# %% [markdown]
# ## Plots for checking data

# %%
# load data
all_data_loaded = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_giant_100hp_varyK inputs_ivvssk/Inputs/RUN_{runid}.pt")
all_label_loaded = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_giant_100hp_varyK inputs_ivvssk/Labels/RUN_{runid}.pt")
print(all_data_loaded.shape, all_label_loaded.shape)
# print(np.where(all_label_loaded.numpy() == all_label_loaded.numpy().max()))
streamlines_faded_calc = all_data_loaded[4]
label_normed = all_label_loaded[0]

# %%
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.imshow(label_normed[1900:2400,2200:2510].T.numpy(),cmap="RdBu_r", interpolation="nearest")
plt.title("Temperature field")
plt.xticks([])
plt.yticks([])
plt.subplot(3,1,2)
plt.imshow(streamlines_faded_real[1900:2400,2200:2510].T.numpy(),cmap="RdBu_r", interpolation="nearest")
plt.title("Calculated streamlines based on true velocities")
plt.xticks([])
plt.yticks([])
plt.subplot(3,1,3)
plt.imshow(streamlines_faded_calc[1900:2400,2200:2510].T, cmap="RdBu_r", interpolation="nearest")
plt.title("Calcuated streamlines based on predicted velocities")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("streamlines_correct_ausschnitt.png")
# plt.show()

# %%
# ref = [1955,  315]
# label = torch.load(destination/"Labels"/f"RUN_{runid}.pt")
# plt.figure()
# plt.subplot(1,4,1)
# plt.imshow(vx_real[ref[0]-10:ref[0]+1000, ref[1]-10:ref[1]+100])
# plt.colorbar()
# plt.subplot(1,4,2)
# plt.imshow(inputs[1][ref[0]-10:ref[0]+1000, ref[1]-10:ref[1]+100])
# plt.colorbar()
# plt.subplot(1,4,3)
# plt.imshow(vy_real[ref[0]-10:ref[0]+1000, ref[1]-10:ref[1]+100])
# plt.colorbar()
# plt.subplot(1,4,4)
# plt.imshow(inputs[2][ref[0]-10:ref[0]+1000, ref[1]-10:ref[1]+100])
# # plt.plot(ref, "ro")
# plt.colorbar()
# plt.tight_layout()
# plt.savefig("testi.png")

# %%
# inputs_tmp = torch.load(destination/"Inputs"/f"RUN_{runid}.pt")
# plt.figure()
# for idx,prop in enumerate(inputs_tmp):
#     plt.subplot(1,len(inputs_tmp), idx+1)
#     plt.imshow(prop.numpy())
#     plt.colorbar()

# plt.savefig("testi2.png")


