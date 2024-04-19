
import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.data_stuff.dataset import (DatasetEncoder, DatasetExtend1,
                                              DatasetExtend2,
                                              SimulationDataset,
                                              SimulationDatasetCuts,
                                              get_splits, random_split_extend)
from utils.utils_data import SettingsTraining

def init_data(settings: SettingsTraining, seed=1):
    if settings.problem == "2stages":
        dataset = SimulationDataset(settings.dataset_prep)
    elif settings.problem == "extend1":
        dataset = DatasetExtend1(settings.dataset_prep, box_size=settings.len_box)
    elif settings.problem == "extend2":
        dataset = DatasetExtend2(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        # dataset = DatasetEncoder(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    elif settings.problem == "allin1":
        if settings.case == "test":
            dataset = SimulationDataset(settings.dataset_prep)
        else:
            dataset = SimulationDatasetCuts(settings.dataset_prep, skip_per_dir=64)
    print(f"Length of dataset: {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # if settings.case == "test": # TODO change back
    #     split_ratios = [0.0, 0.0, 1.0] 
    if not settings.problem == "extend2":
        datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    else:
        datasets = random_split_extend(dataset, get_splits(len(dataset.input_names), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=100, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=100, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=100, shuffle=True, num_workers=0)
    print(len(datasets[0]), len(datasets[1]), len(datasets[2]))

    return dataset.input_channels, dataloaders

def init_data_different_datasets(settings: SettingsTraining, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None):
    dataloaders = {}

    if settings.case == "test":
        dataset = SimulationDataset(settings.dataset_prep)
        dataloaders["test"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
    else:
        dataset = SimulationDatasetCuts(settings.dataset_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box)
        dataloaders["train"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
        if settings_val:
            dataset_tmp = SimulationDatasetCuts(settings_val.dataset_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box)
            dataloaders["val"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)
        if settings_test:
            dataset_tmp = SimulationDataset(settings_test.dataset_prep)
            dataloaders["test"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)

        print(len(dataset), len(dataloaders["val"].dataset), len(dataloaders["test"].dataset))
    return dataset.input_channels, dataloaders

