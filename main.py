import argparse
import logging
import multiprocessing
import numpy as np
import time
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss

from data_stuff.dataset import SimulationDataset, TrainDataset, DatasetExtend1, DatasetExtend2, get_splits
from data_stuff.utils import SettingsTraining, load_yaml
from networks.unet import UNet, UNetBC
from networks.unetHalfPad import UNetHalfPad
from networks.equivariantCNN import G_UNet
from processing.solver import Solver
from processing.rotation import rotate_and_infer
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic
from postprocessing.measurements import measure_loss, save_all_measurements

def init_data(settings: SettingsTraining, seed=1):
    if settings.problem == "2stages":
        dataset = SimulationDataset(settings.dataset_prep)
    elif settings.problem == "extend1":
        dataset = DatasetExtend1(settings.dataset_prep, box_size=settings.len_box)
    elif settings.problem == "extend2":
        dataset = DatasetExtend2(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    print(f"Length of dataset: {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # if settings.case == "test":
    #     split_ratios = [0.0, 0.0, 1.0]
    
    dataset = TrainDataset.restrict_data(dataset, settings.data_n)
    print('------------------------------------------------------------------------------------------------------------------')
    print('Dataset restricted to size ' + str(len(dataset)))
    print('------------------------------------------------------------------------------------------------------------------')

    if settings.rotate_inference and settings.case == 'train':
        dataset = TrainDataset.rotate_data(dataset)

    datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(TrainDataset.augment_data(datasets[0], settings.augmentation_n, settings.mask, settings.rotate_inputs), batch_size=50, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(TrainDataset.augment_data(datasets[1], 0, settings.mask, settings.rotate_inputs), batch_size=50, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(TrainDataset.augment_data(datasets[2], 0, settings.mask, settings.rotate_inputs), batch_size=50, shuffle=True, num_workers=0)

    return dataset.input_channels, dataloaders


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    input_channels, dataloaders = init_data(settings)
    # model
    if settings.problem == "2stages":
        if settings.use_ecnn:
            model = G_UNet(in_channels=input_channels).float()
        else:
            model = UNet(in_channels=input_channels).float()
    elif settings.problem in ["extend1", "extend2"]:
        model = UNetHalfPad(in_channels=input_channels).float()
    if settings.case in ["test", "finetune"]:
        model.load(settings.model, settings.device)
    model.to(settings.device)

    solver = None
    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        finetune = True if settings.case == "finetune" else False
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=finetune, use_ecnn=settings.use_ecnn)
        try:
            solver.load_lr_schedule(settings.destination / "learning_rate_history.csv", settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
            print("Training finished")

    # save model
    model.save(settings.destination)

    # visualization
    which_dataset = "val"
    pic_format = "png"
    times["time_end"] = time.perf_counter()
    if settings.case == "test":
        settings.visualize = True
        which_dataset = "test"
        errors = measure_loss(model, dataloaders, settings, rotate_inference=settings.rotate_inference, mask = True)
        print('----------------------------------------------------------------------------------')
        print(errors)
        print('----------------------------------------------------------------------------------')
    save_all_measurements(settings, len(dataloaders[which_dataset].dataset), times, solver) #, errors)
    if settings.visualize:
        visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", pic_format=pic_format, amount_datapoints_to_visu=10, rotate_inference=settings.rotate_inference, mask=True) #amount_datapoints_to_visu=5,
        times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device, rotate_inference=settings.rotate_inference, mask=True)
        plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
        print("Visualizations finished")
        
    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")
    
    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    if settings.problem == "2stages":
        model = UNet(in_channels=in_channels).float()
    elif settings.problem in ["extend1", "extend2"]:
        model = UNetHalfPad(in_channels=in_channels).float()
    model.load(model_name, settings.device)
    model.eval()

    data_dir = settings.dataset_prep
    (data_dir / "Outputs").mkdir(exist_ok=True)

    for datapoint in (data_dir / "Inputs").iterdir():
        data = torch.load(datapoint)
        data = torch.unsqueeze(data, 0)
        time_start = time.perf_counter()

        if settings.rotate_inference:
            y_out = rotate_and_infer(data, [-1,0], model, load_yaml(settings.destination, 'info'), settings.device).to(settings.device)
        else:
            y_out = model(data.to(settings.device)).to(settings.device)

        time_end = time.perf_counter()
        y_out = y_out.detach().cpu()
        y_out = torch.squeeze(y_out, 0)
        torch.save(y_out, data_dir / "Outputs" / datapoint.name)
        print(f"Inference of {datapoint.name} took {time_end-time_start} seconds")
    
    print(f"Inference finished, outputs saved in {data_dir / 'Outputs'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2d_small_1000dp", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi") #choices=["gki", "gksi", "pksi", "gks", "gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100", "t", "gkiab", "gksiab", "gkt"]
    parser.add_argument("--case_2hp", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--save_inference", type=bool, default=False)
    parser.add_argument("--problem", type=str, choices=["2stages", "allin1", "extend1", "extend2",], default="2stages")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--len_box", type=int, default=256)
    parser.add_argument("--skip_per_dir", type=int, default=256)
    parser.add_argument("--augmentation_n", type=int, default=0)
    parser.add_argument("--rotate_inference", type=bool, default=False)
    parser.add_argument("--use_ecnn", type=bool, default=False)
    parser.add_argument("--mask", type=bool, default=False)
    parser.add_argument("--rotate_inputs", type=int, default=0)
    parser.add_argument("--data_n", type=int, default=-1)
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    settings = prepare_data_and_paths(settings)

    model = run(settings)

    if args.save_inference:
        save_inference(settings.model, len(args.inputs), settings)