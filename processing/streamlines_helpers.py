import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from datetime import datetime
import pathlib

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import load_yaml, save_yaml, make_paths
import preprocessing.preprocessing as prep
from main import read_cla

# functions for data loading + preparing
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

    inputs = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/Inputs/{runid}")
    inputs_normed = deepcopy(inputs)
    norm.reverse(inputs, "Inputs")

    labels = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/Labels/{runid}")
    labels_normed = deepcopy(labels)
    norm.reverse(labels, "Labels")

    args = load_yaml(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/args.yaml")
    print("current:", labels.shape, inputs_normed.shape)

    return inputs, inputs_normed, labels, labels_normed, info, args

def build_new_inputs_and_outputs(inputs_normed, labels_normed, indices):
    dummy_field = torch.zeros_like(inputs_normed[0])

    inputs_new = torch.cat([inputs_normed[indices["mat_id"]].unsqueeze(0), labels_normed[indices["vx"]].unsqueeze(0), labels_normed[indices["vy"]].unsqueeze(0), dummy_field.unsqueeze(0), inputs_normed[indices["k"]].unsqueeze(0)], dim=0) #, dummy_field.unsqueeze(0)
    
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
    "Streamlines Faded [-]",
    "Permeability X [m^2]",
    "Streamlines Faded Outer [-],"
    ]
    args["outputs"] = ["Temperature [C]"]

def build_new_info(info):
    # info["Inputs"]["Streamlines Faded [-]"] = {
    # "index": 3,
    # "max": 1.0,
    # "mean": None,
    # "min": 0.0,
    # "norm": None,
    # "std": None,
    # }
    info["Inputs"]["Streamlines Faded Outer [-]"] = {
    "index": 5,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    # info["Inputs"]["Permeability X [m^2]"]["index"] = 4
    # info["Inputs"]["Liquid X-Velocity [m_per_y]"] = info["Labels"]["Liquid X-Velocity [m_per_y]"]
    # info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"] = 1
    # info["Inputs"]["Liquid Y-Velocity [m_per_y]"] = info["Labels"]["Liquid Y-Velocity [m_per_y]"]
    # info["Inputs"]["Liquid Y-Velocity [m_per_y]"]["index"] = 2

    # info["Labels"] = {"Temperature [C]": info["Labels"]["Temperature [C]"]}

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

# data processing + streamline calculation
def integrate_velocity(x, y, vx, vy):
        fx = RegularGridInterpolator((x,y), vx, bounds_error=False, fill_value=None, method="linear")
        fy = RegularGridInterpolator((x,y), vy, bounds_error=False, fill_value=None, method="linear")

        # define the velocity function to be integrated:
        def f(t, y):
            return np.squeeze([fy(y), fx(y)])

        return f

def calc_streamline(interpolator, maxs_xy, start=[5,14], t_end=27.5, t_steps=1000):
    
    # Solve for start point
    sol = solve_ivp(interpolator, [0, t_end], start, t_eval=np.linspace(0,t_end,t_steps), method="Radau")

    sol_x = sol.y[0]
    sol_y = sol.y[1]

    # cut sol_y, sol_x if extend x.max(),y.max() or x.min(),y.min() (= 0,0)
    sol_x = sol_x[sol_x <= maxs_xy[0]]
    sol_y = sol_y[sol_y <= maxs_xy[1]]
    sol_x = sol_x[sol_x >= 0]
    sol_y = sol_y[sol_y >= 0]

    length = np.min([sol_x.shape[0], sol_y.shape[0]])
    sol_x = sol_x[:length]
    sol_y = sol_y[:length]
    t = sol.t[:length]
    
    return sol_x, sol_y, t

def draw_streamlines(image_data:np.array, streamlines:list, faded:bool=False):
    time = datetime.now()
    for streamline in streamlines:
        if faded:
            val = streamline[2][::-1]
            val = val / val.max()
        else:
            val = 1
        # if np.sum(streamline[0] < 0) > 0 or np.sum(streamline[1] < 0) > 0:
        image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] = val # TODO rm? cell-center offset, because earlier already in there??
    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return image_data

def make_streamlines(mat_ids, vx, vy, dims, offset:str=None):
    pos_hps = np.array(np.where(mat_ids == 2)).T.astype(float)
    print("Number of heat pumps: ", pos_hps.shape[0])
    pos_hps += np.array([0.5,0.5]) # cell-center offset
    resolution = 5
    if offset != None:
        pos_hps += np.array([0,offset])
    x,y = (np.arange(0,dims[0]),np.arange(0,dims[1]))

    time = datetime.now()
    streamlines = []
    integrator = integrate_velocity(x, y, vx/resolution, vy/resolution)
    for hp in tqdm(pos_hps, desc="Calculating streamlines"):
        sol = calc_streamline(integrator, (x.max(),y.max()), hp, t_end=27.5, t_steps=10_000)
        streamlines.append(sol)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")

    streamlines_ohe = draw_streamlines(np.zeros(dims), streamlines, faded=False)
    streamlines_faded = draw_streamlines(np.zeros(dims), streamlines, faded=True)

    return torch.tensor(streamlines_ohe), torch.tensor(streamlines_faded)

# save new data
def save_new_datapoint(destination, runid:str, inputs_new:torch.Tensor, labels_new:torch.Tensor=None):
    (destination/"Inputs").mkdir(exist_ok=True, parents=True)
    (destination/"Labels").mkdir(exist_ok=True, parents=True)
    torch.save(inputs_new, destination / "Inputs"/ runid)
    if labels_new != None:
        torch.save(labels_new, destination / "Labels" / runid)
    correct_args_info(destination)

def correct_args_info(destination):
    info = load_yaml(destination / "info.yaml")
    build_new_info(info)
    print(info)
    save_yaml(info, destination / "info.yaml")

    args = load_yaml(destination / "args.yaml")
    build_new_args(args)
    print(args)
    save_yaml(args, destination / "args.yaml")