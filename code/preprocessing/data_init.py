import os
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch import Generator

from preprocessing.datasets.dataset import DataPoint, DatasetBasis
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts

def init_data(args:dict, tmp_bool_cutouts:bool=False, batchsize:int=64, order_data:list=[0,2,1]):
    dataloaders = {}
    cache_mode = str(args.get("cache", "none")).lower()
    cache_device = args.get("device", "cpu")
    if cache_mode == "cuda" and "cuda" not in str(cache_device):
        cache_mode = "cpu"

    if not tmp_bool_cutouts or args["case"] == "test": # NO CUTOUTS
        if order_data:
            dataset_train = DataPoint(args["data_prep"], i=order_data[0], cache=cache_mode, cache_device=cache_device)
        else: # large dataset with random split, no order_data provided
            dataset = DatasetBasis(args["data_prep"], cache=cache_mode, cache_device=cache_device)
            datasets = random_split(dataset, [0.8, 0.2, 0.0], generator=Generator().manual_seed(1))
            print("ATTENTION!! CHANGED SPLIT to 80:20:0 because hidden test data")
            print(f"Random split of dataset: {len(datasets[0])} training, {len(datasets[1])} validation, {len(datasets[2])} test samples.")
            
    else: # DO CUTOUTS
        if not order_data:
            # load list of names in dir args[data_prep], then split into train/val/test
            # runs = [int(f.split(".pt")[0].split("Sim_")[1]) for f in os.listdir(args["data_prep"]/"Inputs") if f.endswith(".pt")]
            runs = np.arange(len([f for f in os.listdir(args["data_prep"]/"Inputs") if f.endswith(".pt")]))
            np.random.shuffle(runs)
            order_data = runs[:int(len(runs)*0.7)], runs[int(len(runs)*0.7):int(len(runs)*0.9)], runs[int(len(runs)*0.9):]
        dataset_train = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=order_data[0])
        
    if order_data:
        dataset_val = DataPoint(args["data_prep"], i=order_data[1], cache=cache_mode, cache_device=cache_device)
        datasets = [dataset_train, dataset_val]
    else: pass # already defined

    num_workers_default = 0 if cache_mode != "none" else 4
    num_workers = int(args.get("num_workers", num_workers_default))
    if cache_mode == "cuda":
        num_workers = 0
    pin_memory = bool("cuda" in str(args.get("device", "")) and cache_mode != "cuda")
    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, **dataloader_kwargs)
    dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, **dataloader_kwargs)

    # test dataset + dataloader
    if order_data:
        dataset_test = DataPoint(args["data_prep"], i=order_data[2], cache=cache_mode, cache_device=cache_device)
        datasets.append(dataset_test)
    else: pass # already defined
    dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=False, **dataloader_kwargs)

    if order_data:
        return datasets[0].input_channels, datasets[0].output_channels, dataloaders
    else: # due to formality
        return datasets[0].dataset.input_channels, datasets[0].dataset.output_channels, dataloaders
