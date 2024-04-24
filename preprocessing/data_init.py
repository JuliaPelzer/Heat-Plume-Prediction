import argparse
import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.datasets.dataset import get_splits, DataPoint
from preprocessing.datasets.dataset_1stbox import Dataset1stBox
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from preprocessing.datasets.dataset_extend import DatasetExtend, DatasetEncoder, random_split_extend

def init_data(settings: argparse.Namespace, seed=1):
    if settings.problem in ["2stages", "1hp", "test"]:
        dataset = Dataset1stBox(settings.data_prep, box_size=settings.len_box)
    elif settings.problem == "extend":
        dataset = DatasetExtend(settings.data_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        # dataset = DatasetEncoder(settings.data_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    elif settings.problem == "allin1":
        if settings.case == "test":
            dataset_train, dataset_val = None, None
        else:
            dataset_train = SimulationDatasetCuts(settings.data_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box, idx=0)
            dataset_val = SimulationDatasetCuts(settings.data_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box, idx=1)
        dataset_test = DataPoint(settings.data_prep, idx=2)
        #Dataset1stBox(settings.data_prep) # TODO only take idx=2

    split_ratios = [0.7, 0.2, 0.1]
    generator = torch.Generator().manual_seed(seed)
    if settings.problem in ["2stages", "1hp", "test"]:
        datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    elif settings.problem == "extend":
        datasets = random_split_extend(dataset, get_splits(len(dataset.input_names), split_ratios), generator=generator)
    elif settings.problem == "allin1":
        datasets = [dataset_train, dataset_val, dataset_test]

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=100, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=100, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=100, shuffle=True, num_workers=0)

    if not settings.problem == "allin1":
        print(f"Length of dataset: {len(dataset)} - split into {len(datasets[0])}:{len(datasets[1])}:{len(datasets[2])}")
        return dataset.input_channels, dataloaders
    else:
        if settings.case == "test":
            print(f"Length of dataset: {len(dataset_test)}")
        else:
            print(f"Length of dataset: {len(dataset_train)}:{len(dataset_val)}:{len(dataset_test)}")
        return dataset_test.input_channels, dataloaders

def load_all_datasets_as_datapoints(args: argparse.Namespace):
    dataloaders = {}
    for idx, case in zip([0, 1, 2], ["train", "val", "test"]):
        try:
            dataset = DataPoint(args.data_prep, idx=idx)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            dataloaders[case] = dataloader
        except: pass
    return dataloaders    