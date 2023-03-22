import datetime as dt
import logging
import os
import argparse
from dataclasses import dataclass
from networks.models import create_model, load_model
from networks.losses import create_loss_fn
from torch import cuda, save
from torch.utils.tensorboard import SummaryWriter
from data.dataset_loading import init_data, make_dataset_for_test
from data.utils import load_settings, save_settings, SettingsTraining
from solver import Solver
from utils.visualize_data import plot_sample
from utils.utils_networks import count_parameters, append_results_to_csv

def run_training(settings:SettingsTraining):
    time_begin = dt.datetime.now()

    # parameters of data
    sdf = False
    # init data
    _, dataloaders = init_data(settings, batch_size=100, sdf=sdf, labels="t")

    if settings.device is None:
        settings.device = "cuda" if cuda.is_available() else "cpu"
    logging.warning(f"Using {settings.device} device")

    # model choice
    in_channels = len(settings.inputs) + 1
    if not settings.finetune:
        model = create_model(settings.model_choice, in_channels)
    else:
        model = load_model({"model_choice":settings.model_choice, "in_channels":in_channels}, settings.path_to_model)
    model.to(settings.device)

    number_parameter = count_parameters(model)
    logging.warning(f"Model {settings.model_choice} with number of parameters: {number_parameter}")

    train = True
    if train:
        # parameters of training
        loss_fn_str = "MSE"
        loss_fn = create_loss_fn(loss_fn_str, dataloaders)
        # training
        solver = Solver(model,dataloaders["train"], dataloaders["val"], loss_func=loss_fn,)
        try:
            solver.load_lr_schedule(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "learning_rate_history.csv"))
            solver.train(settings)
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt")
    else:
        # load model
        model = load_model({"model_choice":settings.model_choice, "in_channels":in_channels}, os.path.join(os.getcwd(), "runs", settings.name_folder_destination), "model")
        model.to(settings.device)
    # save model
    if train:
        save(model.state_dict(), os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "model.pt"))
        solver.save_lr_schedule(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "learning_rate_history.csv"))
        
    # visualization
    if True:
        error_mean, final_max_error = plot_sample(model, dataloaders["train"], settings.device, plot_name=settings.name_folder_destination + "/plot_train_sample", amount_plots=2,)
        error_mean, final_max_error = plot_sample(model, dataloaders["val"], settings.device, plot_name=settings.name_folder_destination + "/plot_val_sample", amount_plots=10,)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Time needed for experiment: {duration}")

    # logging
    results = {"timestamp": time_begin, "model": settings.model_choice, "dataset": settings.dataset_name, "inputs": settings.inputs, "n_epochs": settings.epochs, "error_mean": error_mean[-1], "error_max": final_max_error, "duration": duration, "name_destination_folder": settings.name_folder_destination,}
    append_results_to_csv(results, "runs/collected_results_rough_idea.csv")

    model.to("cpu")
    return model

def run_tests(settings:SettingsTraining):

    # init data
    _, dataloader = make_dataset_for_test(settings)

    # model choice
    in_channels = len(settings.inputs) + 1
    model = create_model(settings.model_choice, in_channels)
    model.to(settings.device)

    number_parameter = count_parameters(model)
    logging.warning(f"Model {settings.model_choice} with number of parameters: {number_parameter}")

    # load model
    model = load_model({"model_choice":settings.model_choice, "in_channels":in_channels}, os.path.join(os.getcwd(), "runs", settings.name_folder_destination), "model")
    model.to(settings.device)
        
    # visualization
    time_begin = dt.datetime.now()
    plot_sample(model, dataloader, settings.device, plot_name=settings.name_folder_destination + "/plot_TEST_sample", amount_plots=10,)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Time needed for only inference and plotting: {duration}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    remote = True
    parser = argparse.ArgumentParser()
    if remote:
        parser.add_argument("--path_to_datasets", type=str, default="/home/pelzerja/pelzerja/test_nn/datasets")
    else:
        parser.add_argument("--path_to_datasets", type=str, default="/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets")
    
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100dp_vary_hp_loc") # benchmark_dataset_2d_20dp_2hps benchmark_testcases_4 benchmark_dataset_2d_100dp_vary_hp_loc benchmark_dataset_2d_100datapoints dataset3D_100dp_perm_vary dataset3D_100dp_perm_iso
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--epochs", type=int, default=1) #0000)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--path_to_model", type=str, default="current_unet_inputs_pk_moreConvPerBlock_noPowerOf2")
    parser.add_argument("--model_choice", type=str, default="unet")
    parser.add_argument("--inputs", type=str, default="pk") #sm
    parser.add_argument("--name_folder_destination", type=str, default="default")
    args = parser.parse_args()

    settings = SettingsTraining(**vars(args))
    # settings.name_folder_destination = f"current_{settings.model_choice}_inputs_{settings.inputs}" #{kwargs['dataset_name']}_
    
    try:
        os.mkdir(os.path.join(os.getcwd(), "runs", settings.name_folder_destination))
    except FileExistsError:
        pass
    settings.save()
    
    run_training(settings)
    # run_tests(settings)
    #tensorboard --logdir runs/
