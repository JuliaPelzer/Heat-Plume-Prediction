import argparse
import logging
import multiprocessing
import numpy as np
import time
import torch
import yaml
import pathlib
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
from torch.nn import MSELoss, L1Loss
import torchmetrics

from data_stuff.dataset import DatasetExtendConvLSTM, get_splits
from data_stuff.utils import SettingsTraining
from networks.unet import UNet, UNetBC
from networks.unetHalfPad import UNetHalfPad
from networks.convLSTM import Seq2Seq
from processing.solver import Solver
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic, visualizations_convLSTM
from postprocessing.measurements import measure_loss, save_all_measurements

def init_data(settings: SettingsTraining):
    if settings.case == 'test':
        settings.skip_per_dir = 64
    dataset = DatasetExtendConvLSTM(settings.dataset_prep, prev_steps=settings.prev_boxes, extend=settings.extend , skip_per_dir=settings.skip_per_dir, overfit=settings.overfit)
    settings.inputs += "T"
    print(f"Length of dataset: {len(dataset)}")

    split_ratios = [0.7, 0.2, 0.1]
    
    split1, split2, split3 = get_splits(len(dataset), split_ratios)
    datasets = []

    datasets.append(Subset(dataset, range(split3+split2,len(dataset))))
    datasets.append(Subset(dataset, range(split3,split3+split2)))
    datasets.append(Subset(dataset, range(split3)))
    dataloaders = {}
    torch.manual_seed(2809)
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=16, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=16, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=16, shuffle=False, num_workers=0, drop_last=True)

    return dataset.input_channels, dataloaders


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    input_channels, dataloaders = init_data(settings)
    # model
    model = Seq2Seq(in_channels=input_channels, frame_size=(64,64), prev_boxes = settings.prev_boxes, extend=settings.extend, num_layers=settings.num_layers,
    enc_conv_features=settings.enc_conv_features,
    dec_conv_features=settings.dec_conv_features,
    enc_kernel_sizes=settings.enc_kernel_sizes,
    dec_kernel_sizes=settings.dec_kernel_sizes).float()
    if settings.case in ["test", "finetune", "benchmark"]:
        model.load(settings.model, settings.device)
    model.to(settings.device)

    solver = None
    if settings.case in ["train", "finetune"]:
        if settings.loss == 'mse':
            loss_fn = MSELoss()
        if settings.loss == 'l1':
            loss_fn = L1Loss()

        # training
        finetune = True if settings.case == "finetune" else False
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=finetune)
        try:
            solver.load_lr_schedule(settings.destination / "learning_rate_history.csv", settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            model.save(settings.destination)
            print(f"Model saved in {settings.destination}")
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
            print("Training finished")

    # save model
    model.save(settings.destination)
    print(f"Model saved in {settings.destination}")

    # number of model parameters
    num_param = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_param}")

    # visualization
    which_dataset = "val"
    pic_format = "png"
    times["time_end"] = time.perf_counter()
    
    settings.visualize == True
    if settings.case == "test":
        settings.visualize = True
        which_dataset = "test"
        dp_to_visu = [(20 - settings.extend - settings.prev_boxes +1) * i for i in range(10)]
    benchmark=False
    if settings.case == "benchmark":
        benchmark = True
        which_dataset = "test"
    errors = measure_loss(model, dataloaders[which_dataset], device="cuda:0", benchmark=benchmark)
    save_all_measurements(settings, dataloaders[which_dataset], times, solver, errors)
    if settings.visualize:
        visualizations_convLSTM(model, dataloaders['test'], settings.device, prev_boxes=settings.prev_boxes, extend=settings.extend, plot_path=settings.destination, dp_to_visu=dp_to_visu, pic_format=pic_format)
        #times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
        #plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
            
        print("Visualizations finished")
        
    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    model = Seq2Seq(in_channels=in_channels, frame_size=(64,64), prev_boxes = settings.prev_boxes, extend=settings.extend, num_layers=settings.num_layers,
    enc_conv_features=settings.enc_conv_features,
    dec_conv_features=settings.dec_conv_features,
    enc_kernel_sizes=settings.enc_kernel_sizes,
    dec_kernel_sizes=settings.dec_kernel_sizes).float()
    model.load(model_name, settings.device)
    model.eval()

    data_dir = settings.dataset_prep
    input_dir = pathlib.Path(f'{data_dir}/Inputs')
    output_dir = pathlib.Path(f'{data_dir}/Outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    for datapoint in input_dir.iterdir():
        data = torch.load(datapoint)
        data = torch.unsqueeze(data, 0)
        print(f"Shape of data: {data.shape}")
        time_start = time.perf_counter()
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
    parser.add_argument("--dataset_raw", type=str, default="ep_medium_1000dp_only_vary_dist", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="ep_medium_1000dp_only_vary_dist inputs_ks")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune", "benchmark"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="ks", choices=["ks", "gki", "gksi", "gks"])
    parser.add_argument("--problem", type=str, default="extend2")
    parser.add_argument("--prev_boxes", type=int, default=1)
    parser.add_argument("--extend", type=int, default=2)
    parser.add_argument("--overfit", type=int, default=0, help="Amount of datapoints the model overfits to. In case of 0, the model considers all datapoints.")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--loss", type=str, choices=['mse', 'l1'], default='mse')
    parser.add_argument("--enc_conv_features", default=[16, 32, 64])
    parser.add_argument("--dec_conv_features", default=[64,32,16,8])
    parser.add_argument("--enc_kernel_sizes", default = [5, 5, 5, 5])
    parser.add_argument("--dec_kernel_sizes", default=[5, 5, 5, 5])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--skip_per_dir", type=int, default=64)
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    settings = prepare_data_and_paths(settings)

    model = run(settings)

    # if args.save_inference:
    #     save_inference(settings.destination, len(args.inputs)+1, settings)