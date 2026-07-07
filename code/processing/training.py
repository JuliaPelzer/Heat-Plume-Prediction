import logging
import multiprocessing
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
from datetime import datetime
from typing import Dict
from pathlib import Path

from preprocessing.preprocessing import preprocessing
from preprocessing.data_init import init_data
from processing.networks.unetVariants import UNetNoPad2, UNet
from processing.solver import Solver
from processing.loss_fcts import CombiLoss, WeightedMSELoss
from postprocessing.visualization import visualizations
from utils.utils_args import load_yaml, save_yaml, check_model_avail, load_hyperparams

def training(args: Dict):
    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    args = load_hyperparams(args)
    save_yaml(args, args["destination"] / "command_line_arguments.yaml")
    preprocessing(args) # and save info.yaml in model folder
    
    input_channels, output_channels, dataloaders = init_data(args, batchsize=args["batchsize"], tmp_bool_cutouts=args["bool_cutouts"], order_data=args["order_data"])
    
    # if (input_channels == 3 and output_channels == 1) or output_channels > 1:
    model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
    # else:
    #     model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
    model.to(args["device"])
    print(f"Model has {model.num_of_params()} parameters")
    
    if args["case"] in ["test", "finetune"]:
        check_model_avail(args)
        model.load(args["destination"], args["device"])
    if args["case"] == "test":
        model.eval()

    if args["case"] in ["train", "finetune"]:
        loss = select_loss_function(args)
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"), learning_rate=args["lr"])
        training_time = datetime.now()
        try:
            solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
            print("Training finished")

        # save model 
        training_time = datetime.now() - training_time
        model.load_state_dict(solver.best_model_params["state_dict"])
        model.save(args["destination"])
        solver.save_metrics_separate_yaml(args["destination"], model.num_of_params(), args["epochs"], training_time.total_seconds(), args["device"])

    # for case, dataloader in dataloaders.items():
    #     for inputs, labels in dataloader:
    #         print(inputs.shape, labels.shape, "shape of inputs and labels")
    #         len_batch = inputs.shape[0]
    #         for datapoint_id in range(len_batch):
    #             x = inputs[datapoint_id]
    #             y_out = model.infer(x.unsqueeze(0), args["device"])
    #             torch.save(y_out, f"{args['destination']}/prediction_{case}.pt")

    # postprocessing
    for case in ["train"]: #, "val", "test"]:
        visualizations(model, dataloaders[case], args, plot_path=args["destination"] / case, amount_datapoints_to_visu=1, pic_format="png")

    return model

def run(trial, args: Dict, PATH_DATA_PREP: Path):
    # TODO outdated
    config = load_yaml(args["destination"] / "HPS_options.yaml")
    (args["destination"] / "models").mkdir(parents=True, exist_ok=True)

    run_name = trial.number
    args["inputs"] = trial.suggest_categorical("inputs", config["inputs"]["values"])
    args["len_box"] = trial.suggest_categorical("len_box", config["len_box"]["values"])
    args["skip_per_dir"] = trial.suggest_categorical("skip_per_dir", config["skip_per_dir"]["values"])
    args["stride"] = trial.suggest_categorical("stride", config["stride"]["values"])
    args["dilation"] = trial.suggest_categorical("dilation", config["dilation"]["values"])
    args["activation_fct"] = trial.suggest_categorical("activation_fct", config["activation_fct"]["values"])
    args["norm"] = trial.suggest_categorical("norm", config["norm"]["values"])
    args["repeat_inner"] = trial.suggest_categorical("repeat_inner", config["repeat_inner"]["values"])
    args["bool_cutouts"] = trial.suggest_categorical("bool_cutouts", config["bool_cutouts"]["values"])
    args["batchsize"] = trial.suggest_categorical("batchsize", config["batchsize"]["values"])
    args["depth"] = trial.suggest_categorical("depth", config["depth"]["values"])
    args["init_features"] = trial.suggest_categorical("init_features", config["init_features"]["values"])
    args["kernel_size"] = trial.suggest_categorical("kernel_size", config["kernel_size"]["values"])
    args["lr"] = float(trial.suggest_categorical("lr", config["lr"]["values"]))
    args["train_loss"] = trial.suggest_categorical("train_loss", config["train_loss"]["values"])

    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)
    if len(args["outputs"]) == 2:
        # TODO: change name of dataset
        args["data_prep"] = PATH_DATA_PREP / f"dataset_giant_100hp_varyK inputs_{args['inputs']} outputs_xy"
        # args["data_prep"] = PATH_DATA_PREP / f"dataset_100hp_giant_real_fixP0_0025 inputs_{args['inputs']} outputs_xy"
    elif len(args["outputs"]) == 1:
        # experiment on the inputs of step 3 not included in the automated tests, but explicitly tested
        print(args["data_prep"])

    save_yaml(args, args["destination"] / "command_line_arguments.yaml")

    # data
    preprocessing(args) # and save info.yaml in model folder
    input_channels, output_channels, dataloaders = init_data(args, tmp_bool_cutouts=args["bool_cutouts"], batchsize=args["batchsize"], order_data=args["order_data"])

    try:
        # model
        model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
        model.to(args["device"])
        
        if args["case"] in ["test", "finetune"]:
            check_model_avail(args)
            model.load(args["model"], args["device"])
        if args["case"] == "test":
            model.eval()

        if args["case"] in ["train", "finetune"]:
            loss_mapping = {
                "mae": L1Loss(),
                "mse": MSELoss()
            }
            loss = loss_mapping.get(args["train_loss"].lower(), MSELoss())
            solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"), learning_rate=float(args["lr"]))
            try:
                solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
                val_loss = solver.train(args, optuna_trial=trial)
            except KeyboardInterrupt:
                logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
                val_loss = solver.best_model_params["loss"]

            # save model 
            model.save(args["destination"] / "models", f"{run_name}.pt")

    except Exception as e:
        print(f"An error occurred: {e}")
        val_loss = 0.2

    # Clear up memory
    del model
    del dataloaders
    torch.cuda.empty_cache()

    return val_loss

def select_loss_function(args):
    if args["train_loss"].lower() == "mae":
        loss = L1Loss()
    elif args["train_loss"].lower() == "mse":
        loss = MSELoss()
    elif args["train_loss"].lower() == "weightedmse":
        loss = WeightedMSELoss()
    elif args["train_loss"].lower() == "huber":
        loss = HuberLoss()
    elif args["train_loss"].lower() == "combi":
        loss = CombiLoss(0.75)
    return loss
