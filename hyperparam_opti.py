import argparse
from datetime import datetime
import os
from pathlib import Path
import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState

from torch.nn import MSELoss
from processing.solver import Solver
import utils.utils_args as ut
import preprocessing.preprocessing as prep
from processing.networks.unetVariants import UNetNoPad2
from main import read_cla
from preprocessing.data_init import init_data, load_all_datasets_in_full
from postprocessing.visualization import visualizations


def define_model(trial, input_channels, output_channels):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    depth = trial.suggest_int("depth", 1, 3)
    init_features = trial.suggest_categorical("init_features", [8, 16, 32, 64, 128])
    kernel_size = trial.suggest_int("kernel_size", 3, 5)
    activation = trial.suggest_categorical("activation", ["ReLU"]) #, "LeakyReLU"])

    return UNetNoPad2(in_channels=input_channels, out_channels=output_channels, init_features=init_features, depth=depth, kernel_size=kernel_size, activation_fct = activation).float()

def objective_velo(trial):

    args["inputs"] = trial.suggest_categorical("input_vars", ["pk", "ki"]) # "pki"
    args["skip_per_dir"] = trial.suggest_categorical("skip_per_dir", [8, 16, 32, 64])
    args["len_box"] = trial.suggest_categorical("len_box", [64, 128, 256, 512])

    return objective(trial, args)

def objective_Temp(trial):

    # args["inputs"] = trial.suggest_categorical("input_vars", ["ixycdk", "ixyck", "ixydk", "icdk", "xycdk", "ixycd", "xycd"])
    args["inputs"] = trial.suggest_categorical("input_vars", ["xydk"])
    args["skip_per_dir"] = trial.suggest_categorical("skip_per_dir", [8, 16, 32, 64])
    args["len_box"] = trial.suggest_categorical("len_box", [64, 128, 256, 512])

    return objective(trial, args)

def objective(trial, args: dict):

    args["data_prep"] = None # !!
    ut.make_paths(args) # and check if data / model exists
    ut.save_yaml(args, args["destination"] / "command_line_arguments.yaml")
    
    # prepare data
    prep.preprocessing(args) # and save info.yaml in model folder

    # Get the dataset.
    input_channels, output_channels, dataloaders = init_data(args)

    # Generate the model.
    model = define_model(trial, input_channels, output_channels)
    model.to(args["device"])

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=1e-4)

    # Training of the model.
    solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=MSELoss(), opt=optimizer, learning_rate=1e-4)
    try:
        training_time = datetime.now()
        solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
        loss = solver.train(trial, args)
        training_time = datetime.now() - training_time
        solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
        model.save(args["destination"], model_name = f"model_trial_{trial.number}.pt")
        solver.save_metrics(args["destination"], model.num_of_params(), args["epochs"], training_time, args["device"])

        dataloader = load_all_datasets_in_full(args)["val"]
        visualizations(model, dataloader, args, plot_path=args["destination"] / f"trial{trial.number}_val", amount_datapoints_to_visu=1, pic_format="png")
    except Exception as e:
        print(f"Training failed with exception: {e}")
        loss = 1

    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="allin1")
    parser.add_argument("--data_raw", type=str, default="dataset_giant_100hp_varyK")
    parser.add_argument("--data_prep", type=str, default=None)
    parser.add_argument("--allin1_prepro_n_case", type=str, default=None)
    parser.add_argument("--inputs", type=str, default="pki")
    parser.add_argument("--outputs", type=str, default="t")
    parser.add_argument("--len_box", type=int, default=256)
    parser.add_argument("--skip_per_dir", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--case", type=str, default="train")
    parser.add_argument("--model", type=str, default=None) # required for testing or finetuning
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--device", type=str, default="3", help="cuda device number or 'cpu'")
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args["device"] if not args["device"]=="cpu" else "" #e.g. "1"
    args["device"] = torch.device(f"cuda:{args['device']}" if not args["device"]=="cpu" else "cpu")
    DEVICE = args["device"]

    args["destination"] = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1") / args["destination"]
    current_destination = args["destination"]
    args = read_cla(args["destination"])
    args["destination"] = current_destination # just to make sure that nothing is overwritten
    
    study = optuna.create_study(direction="minimize", storage=f"sqlite:///runs/allin1/{args['destination'].name}/trials.db", study_name="allin1_Temp", load_if_exists=True)
    study.optimize(objective_Temp, n_trials=25)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    print("Done")