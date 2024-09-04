from functools import partial
from networks.unet import UNet, UNetBC
from networks.turbnet import TurbNetG
from networks.unetParallel import UNetParallel
from networks.unetRectKernel import UNetRectKernel
import torch
from torch.nn import Module, MSELoss, modules
from torch.optim import Adam, Optimizer, RMSprop
from pathlib import Path
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
import tempfile
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
import os
from torch.utils.tensorboard import SummaryWriter
from torch import manual_seed
from tqdm.auto import tqdm
from data_stuff.utils import SettingsTraining
import csv
from torch.utils.data import DataLoader
from ray.train import RunConfig
from ray.tune.search.optuna import OptunaSearch
from data_stuff.dataset import SimulationDataset, DatasetExtend1, DatasetExtend2, get_splits

def tune_nn(settings: SettingsTraining, num_samples=200, max_num_epochs=88, gpus_per_trial=1, ):
    if settings.problem == "turbnet":
        config = {
            "features_exp": tune.choice(range(2,8)),
            "lr": tune.choice(1e-5,7e-4,5e-4,3e-4,1e-4,7e-3,5e-3,3e-3, 1e-3),
            "dropout": tune.choice([0.0,0.1,0.2,0.3,0.4,0.5]),
            "weight_decay": tune.choice(1e-5,7e-4,5e-4,3e-4,1e-4,7e-3,5e-3,3e-3, 1e-3),
        }
    elif settings.problem == "parallel":
        kernel_sizes = [(4*i,i) for i in range(2,5)]
        kernel_sizes = kernel_sizes + [(2*j,j) for j in range(2,5)]
        config = {
            "features": tune.choice([32,64]),
            "lr": tune.choice([1e-4]),
            "depth": tune.choice([5]),
            "kernel_size": tune.choice([5]),
            "weight_decay": tune.choice([1e-5]),
            "padding_mode": tune.choice(['replicate']),
            "par_depth": tune.choice([1,2,3]),
            "par_dil": tune.choice([(6,1),(4,1),(2,1),(1,1),(6,2),(4,2),(2,2)]),
            "par_kern": tune.choice([(4,1),(8,2),(12,3),(16,4),(2,1),(4,2),(8,4),(16,8),(1,4),(2,8),(3,12)]),
        }
    elif settings.problem == "rect":
        config = {
            "features": tune.choice([2**i for i in range(4,7)]),
            "lr": tune.choice([1e-4]),
            "depth": tune.choice([2,3,4,5]),
            "kernel_size": tune.choice([3,4,5,6,7]),
            "weight_decay": tune.choice([1e-5]),
            "padding_mode": tune.choice(['zeros','replicate']),
            "dilation": tune.choice([1,2,3]),
            "down_kernel": tune.choice([3,5,7,9,11])
        }
    else:
        config = {
            "features": tune.choice([2**i for i in range(4,7)]),
            "lr": tune.choice([1e-4]),
            "depth": tune.choice([2,3,4,5]),
            "kernel_size": tune.choice([3,4,5,6,7]),
            "weight_decay": tune.choice([1e-5]),
            "padding_mode": tune.choice(['zeros','replicate']),
            "dilation": tune.choice([1,2,3])
        }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    algo = OptunaSearch()
    trainable_with_resources = tune.with_resources(partial(train_mnist, settings=settings), {"cpu": 4, "gpu": gpus_per_trial})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            max_concurrent_trials=1,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": max_num_epochs},
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)
    print("path is:", results.get_best_result().path)
    print("metrics are", results.get_best_result().metrics_dataframe)
    # result = tune.run(
    #     partial(train_mnist, settings=settings),
    #     resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     search_alg=algo,
    # )

    #best_trial = results.get_best_trial("loss", "min", "last")
    #print(f"Best trial config: {best_trial.config}")
    #print(f"Best trial final validation loss: {best_trial.last_result['loss']}")


def train_mnist(config,settings=None):
    torch.cuda.empty_cache()
    input_channels, dataloaders = init_data(settings)
    if settings.problem == "turbnet":
        model = TurbNetG(in_channels=input_channels,channelExponent=config["features_exp"],dropout=config["dropout"]).float()
    elif settings.problem == "parallel":
        model = UNetParallel(in_channels=input_channels,
                            init_features=config["features"],
                            depth=config["depth"],
                            padding_mode=config["padding_mode"],
                            dilation=1,
                            par_depth=config["par_depth"],
                            par_dil=config["par_dil"],
                            par_kern=config["par_kern"]).to(settings.device)
    elif settings.problem == "rect":
        model = UNetRectKernel(in_channels=input_channels,init_features=config["features"],depth=config["depth"],padding_mode=config["padding_mode"],dilation=config["dilation"],down_kernel=config["down_kernel"]).to(settings.device)
    else:
        model = UNet(in_channels=input_channels,init_features=config["features"],depth=config["depth"],padding_mode=config["padding_mode"],dilation=config["dilation"]).to(settings.device)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_function = MSELoss()
    device = settings.device
    model = model.to(device)

    train_dataloader = dataloaders["train"]

    epochs = tqdm(range(settings.epochs), desc="epochs", disable=False)
    while True:

        # Training
        model.train()
        train_loss = 0.0
        for x, y,fname in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = None
            loss =  loss_function(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
        train_loss /= len(train_dataloader)
    
        loss = test_loss(model, dataloaders["test"], settings.device)
        train.report({"loss": loss})


        


def test_loss(model: UNet, dataloader: DataLoader, device: str, loss_func: modules.loss._Loss = MSELoss()):
    model.eval()
    mse_loss = 0.0

    for x, y, fname in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x).to(device)
        mse_loss += loss_func(y_pred, y).detach().item()
        
    mse_loss /= len(dataloader)

    return mse_loss


def init_data(settings: SettingsTraining, seed=1):
    if settings.problem in ["2stages", "turbnet","parallel","rect"]:
        dataset = SimulationDataset(settings.dataset_prep)
    elif settings.problem == "extend1":
        dataset = DatasetExtend1(settings.dataset_prep, box_size=settings.len_box)
    elif settings.problem == "extend2":
        dataset = DatasetExtend2(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    print(f"Length of dataset: {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    if settings.case == "test":
        split_ratios = [0.0, 0.0, 1.0] 

    datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=20, shuffle=True, num_workers=1)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=20, shuffle=True, num_workers=1)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=20, shuffle=True, num_workers=1)

    return dataset.input_channels, dataloaders
