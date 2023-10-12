import argparse
import logging
import os

import torch
from torch import save
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir=runs/ --host localhost --port 8088

from data_stuff.dataset import SimulationDataset, _get_splits
from data_stuff.utils import SettingsTraining
from networks.unet import UNet
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from utils.prepare_paths import set_paths_1hpnn, Paths1HP
from utils.visualization import plt_avg_error_cellwise, plot_sample
from utils.measurements import measure_loss, save_all_measurements


def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(os.path.join(settings.datasets_dir, settings.dataset_prep))
    generator = torch.Generator().manual_seed(seed)

    # split_ratios = [0.7, 0.2, 0.1] # für Training etc.
    split_ratios = [0.0, 0.0, 1.0] # falls wirklich nur Inference
    datasets = random_split(dataset, _get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)

    return dataset, dataloaders


def run(settings: SettingsTraining):

    dataset, dataloaders = init_data(settings)

    # init, load and save model
    model = UNet(in_channels=dataset.input_channels).float()
    model.load_state_dict(torch.load(f"{settings.model}/model.pt", map_location=torch.device(settings.device)))
    model.to(settings.device)
    save(model.state_dict(), os.path.join(os.getcwd(), "runs", settings.destination_dir, "model.pt"))

    # visualization
    if settings.visualize:
        plot_sample(model, dataloaders["test"], settings.device, plot_name=settings.destination_dir + "/plot_test", amount_plots=5, pic_format="png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="benchmark_dataset_2d_100datapoints", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--model", type=str, default="default") #for finetuning or testing
    parser.add_argument("--destination_dir", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi")
    parser.add_argument("--visualize", type=bool, default=False)
    settings = SettingsTraining(**vars(parser.parse_args()))
    settings.case = "test"

    paths: Paths1HP
    paths, dataset_prep = set_paths_1hpnn(settings.dataset_raw, settings.inputs)
    settings.datasets_dir = paths.datasets_prepared_dir
    settings.dataset_prep = dataset_prep

    # prepare dataset if test=case do it anyways because of potentially different std,mean,... values than trained with
    prepare_dataset_for_1st_stage(paths, settings)
    print(f"Dataset {paths.dataset_1st_prep_path} prepared")

    settings.save()

    run(settings)