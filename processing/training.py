import argparse
import logging
import multiprocessing
import time
from copy import deepcopy

import numpy as np
import torch
from torch.nn import MSELoss

from postprocessing.measurements import (save_all_measurements)
from postprocessing.visualization import (infer_all_and_summed_pic,
                                          plot_avg_error_cellwise,
                                          visualizations)
from processing.networks.encoder import Encoder
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad2
from processing.solver import Solver
from utils.utils_data import SettingsTraining
from preprocessing.data_init import init_data, init_data_different_datasets

def train(args: argparse.Namespace, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None, different_datasets: bool = False):
    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    if args.problem == "allin1":
        input_channels, dataloaders = init_data_different_datasets(args, settings_val, settings_test)
    else:
        input_channels, dataloaders = init_data(args)

    # model
    if args.problem in ["2stages", "allin1", "extend1"]:
        model = UNet(in_channels=input_channels).float()
    elif args.problem in ["extend2"]:
        model = UNetHalfPad2(in_channels=input_channels).float()
        # model = Encoder(in_channels=input_channels).float()

    if args.case in ["test", "finetune"]:
        model.load(args.model, args.device)
    model.to(args.device)

    if args.case in ["train", "finetune"]:
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=MSELoss(), finetune=(args.case == "finetune"))
        try:
            solver.load_lr_schedule(args.destination / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args.destination / "learning_rate_history.csv")
            print("Training finished")
    else:
        solver = None

    # save model
    model.save(args.destination)

    # visualization
    save_all_measurements(args, len(dataloaders["val"].dataset), times={}, solver=solver) #, errors)
    try:
        visualizations(model, dataloaders["val"], args, plot_path=args.destination / f"val", amount_datapoints_to_visu=1, pic_format="png", different_datasets=different_datasets)
    except: pass
    visualizations(model, dataloaders["test"], args, plot_path=args.destination / f"test", amount_datapoints_to_visu=1, pic_format="png", different_datasets=different_datasets)

    return model
