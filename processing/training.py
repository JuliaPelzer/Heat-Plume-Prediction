import argparse
import logging
import multiprocessing
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss

# from postprocessing.measurements import (save_all_measurements)
from postprocessing.visualization import (infer_all_and_summed_pic, plot_avg_error_cellwise, visualizations)
from processing.networks.encoder import Encoder
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad2, UNetNoPad2
from processing.solver import Solver
from preprocessing.data_init import init_data, load_all_datasets_as_datapoints

def train(args: dict):
    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    input_channels, dataloaders = init_data(args)

    # model
    if args["problem"] in ["1hp", "2stages", "test"]:
        model = UNet(in_channels=input_channels).float()
    elif args["problem"] in ["extend"]:
        model = UNetHalfPad2(in_channels=input_channels).float()
        # model = Encoder(in_channels=input_channels).float()
    elif args["problem"] in ["allin1"]:
        model = UNet(in_channels=input_channels).float()
        # model = UNetNoPad2(in_channels=input_channels).float()
    model.to(args["device"])

    if args["case"] in ["test", "finetune"]:
        model.load(args["model"], args["device"])

    if args["case"] in ["train", "finetune"]:
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=L1Loss(), finetune=(args["case"] == "finetune"))
        try:
            solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
            print("Training finished")
    else:
        solver = None

    # save model
    if args["case"] in ["train", "finetune"]:
        model.save(args["destination"])

    # postprocessing
    # save_all_measurements(args, len(dataloaders["val"].dataset), times={}, solver=solver) #, errors)
    if args["problem"] == "allin1":
        dataloaders = load_all_datasets_as_datapoints(args)
    for case in ["test", "val", "train"]:
        # try:
        visualizations(model, dataloaders[case], args, plot_path=args["destination"] / case, amount_datapoints_to_visu=1, pic_format="png")
        # except: pass

    return model