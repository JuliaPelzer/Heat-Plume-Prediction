import argparse
import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.datasets.dataset import get_splits
from preprocessing.datasets.dataset_1stbox import Dataset1stBox
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from preprocessing.datasets.dataset_extend import DatasetExtend, DatasetEncoder, random_split_extend
from utils.utils_data import SettingsTraining

def init_data(settings: argparse.Namespace, seed=1):
    if settings.problem in ["2stages", "1hp", "test"]:
        dataset = Dataset1stBox(settings.data_prep, box_size=settings.len_box)
    elif settings.problem == "extend":
        dataset = DatasetExtend(settings.data_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        # dataset = DatasetEncoder(settings.data_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    elif settings.problem == "allin1":
        if settings.case == "test":
            dataset = Dataset1stBox(settings.data_prep)
        else:
            dataset = SimulationDatasetCuts(settings.data_prep, skip_per_dir=64)
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    if not settings.problem == "extend":
        datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    else:
        datasets = random_split_extend(dataset, get_splits(len(dataset.input_names), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=100, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=100, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=100, shuffle=True, num_workers=0)
    print(f"Length of dataset: {len(dataset)} - split into {len(datasets[0])}:{len(datasets[1])}:{len(datasets[2])}")

    return dataset.input_channels, dataloaders

def init_data_different_datasets(settings: argparse.Namespace, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None):
    dataloaders = {}

    if settings.case == "test":
        dataset = Dataset1stBox(settings.data_prep)
        dataloaders["test"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
    else:
        dataset = SimulationDatasetCuts(settings.data_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box)
        dataloaders["train"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
        if settings_val:
            dataset_tmp = SimulationDatasetCuts(settings_val.data_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box)
            dataloaders["val"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)
        if settings_test:
            dataset_tmp = Dataset1stBox(settings_test.data_prep)
            dataloaders["test"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)

        print(len(dataset), len(dataloaders["val"].dataset), len(dataloaders["test"].dataset))
    return dataset.input_channels, dataloaders

