import argparse
import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.datasets.dataset import get_splits, DataPoint
from preprocessing.datasets.dataset_1stbox import Dataset1stBox
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from preprocessing.datasets.dataset_extend import DatasetExtend, DatasetEncoder, random_split_extend

def init_data(args:dict, seed=1):
    if args["problem"] in ["2stages", "1hp", "test"]:
        dataset = Dataset1stBox(args["data_prep"], box_size=args["len_box"])
    elif args["problem"] == "extend":
        dataset = DatasetExtend(args["data_prep"], box_size=args["len_box"], skip_per_dir=args["skip_per_dir"])
        # dataset = DatasetEncoder(args["data_prep"], box_size=args["len_box"], skip_per_dir=args["skip_per_dir"])
        args["inputs"] += "T"
    elif args["problem"] == "allin1":
        if args["case"] == "test":
            dataset_train = DataPoint(args["data_prep"], idx=0)
            dataset_val = DataPoint(args["data_prep"], idx=1)
        else:
            dataset_train = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], idx=0)
            dataset_val = DataPoint(args["data_prep"], idx=1)
            # dataset_val = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], idx=1)

        dataset_test = DataPoint(args["data_prep"], idx=2)

    split_ratios = [0.7, 0.2, 0.1]
    generator = torch.Generator().manual_seed(seed)
    if args["problem"] in ["2stages", "1hp", "test"]:
        datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    elif args["problem"] == "extend":
        datasets = random_split_extend(dataset, get_splits(len(dataset.input_names), split_ratios), generator=generator)
    elif args["problem"] == "allin1":
        datasets = [dataset_train, dataset_val, dataset_test]

    dataloaders = {}
    batchsize = 20
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=True, num_workers=0)

    if not args["problem"] == "allin1":
        print(f"Length of dataset: {len(dataset)} - split into {len(datasets[0])}:{len(datasets[1])}:{len(datasets[2])}")
        return dataset.input_channels, dataset.output_channels, dataloaders
    else:
        try:
            print(f"Length of dataset: {len(dataset_train)}:{len(dataset_val)}:{len(dataset_test)}")
        except:
            print(f"Length of dataset: {len(dataset_test)}")
        return dataset_test.input_channels, dataset_test.output_channels, dataloaders

def load_all_datasets_in_full(args: dict):
    dataloaders = {}
    for idx, case in zip([0, 1, 2], ["train", "val", "test"]):
        if args["problem"] == "allin1":
            try:
                dataset = DataPoint(args["data_prep"], idx=idx)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                dataloaders[case] = dataloader
            except: pass
        else:
            dataset = Dataset1stBox(args["data_prep"])
            dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0)
            dataloaders[case] = dataloader
    return dataloaders    