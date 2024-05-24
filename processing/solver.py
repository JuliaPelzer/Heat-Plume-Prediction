import csv
import logging
import pathlib
import time
from dataclasses import dataclass

from torch.nn import Module, MSELoss, modules
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import manual_seed
from tqdm.auto import tqdm
from postprocessing.visualization import visualizations

from data_stuff.utils import SettingsTraining
from networks.unet import weights_init, UNet
from networks.unetHalfPad import UNetHalfPad

@dataclass
class Solver(object):
    model: Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    loss_func: modules.loss._Loss = MSELoss()
    learning_rate: float = 1e-5
    opt: Optimizer = Adam
    finetune: bool = False
    best_model_params: dict = None

    def __post_init__(self):
        self.opt = self.opt(self.model.parameters(),
                            self.learning_rate, weight_decay=1e-4)
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}

        if not self.finetune:
            self.model.apply(weights_init)

    def train(self, settings: SettingsTraining):
        manual_seed(0)
        log_val_epoch = True
        if log_val_epoch:
            file = open(settings.destination / "log_loss_per_epoch.csv", 'w', newline='')
            csv_writer = csv.writer(file)
            csv_writer.writerow(["epoch", "val loss", "train loss"])
            file = open(settings.destination / "log_best_loss_per_epoch.csv", 'w', newline='')
            csv_writer_best = csv.writer(file)
            csv_writer_best.writerow(["epoch", "val loss", "train loss"])

        start_time = time.perf_counter()
        # initialize tensorboard
        writer = SummaryWriter(settings.destination)
        device = settings.device
        self.model = self.model.to(device)
        writer.add_graph(self.model, next(iter(self.train_dataloader))[0].to(device))

        epochs = tqdm(range(settings.epochs), desc="epochs", disable=False)
        for epoch in epochs:
            try:
                # Set lr according to schedule
                if epoch in self.lr_schedule.keys():
                    self.opt.param_groups[0]["lr"] = self.lr_schedule[epoch]

                # Training
                self.model.train()
                train_epoch_loss = self.run_epoch(
                    self.train_dataloader, device)

                # Validation
                self.model.eval()
                val_epoch_loss = self.run_epoch(self.val_dataloader, device)

                # Logging
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar(
                    "learning_rate", self.opt.param_groups[0]["lr"], epoch)
                epochs.set_postfix_str(
                    f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")
                
                # Keep best model
                if self.best_model_params is None or val_epoch_loss < self.best_model_params["loss"]:
                    self.best_model_params = {
                        "epoch": epoch,
                        "loss": val_epoch_loss,
                        "train loss": train_epoch_loss,
                        "val RMSE": val_epoch_loss**0.5, # TODO only true if loss_func == MSELoss()
                        "train RMSE": train_epoch_loss**0.5, #TODO only true if loss_func == MSELoss()
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }

                if log_val_epoch:
                    if epoch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000, 15000, 20000, 24999]:
                        csv_writer.writerow([epoch, val_epoch_loss, train_epoch_loss])
                        csv_writer_best.writerow([epoch, self.best_model_params["loss"], self.best_model_params["train loss"]])
                        # for name, param in self.model.named_parameters():
                        #     writer.add_histogram(name, param, epoch)

            except KeyboardInterrupt:
                model_tmp = UNetHalfPad(in_channels=len(settings.inputs), out_channels=1) # UNet
                model_tmp.load_state_dict(self.best_model_params["state_dict"])
                model_tmp.to(settings.device)
                model_tmp.save(settings.destination, model_name=f"interim_model_e{epoch}.pt")
                visualizations(model_tmp, self.val_dataloader, settings.device, plot_path=settings.destination / f"plot_val_interim_e{epoch}", amount_datapoints_to_visu=2, pic_format="png")

                try:
                    new_lr = float(input("\nNew learning rate: "))
                except ValueError as e:
                    print(e)
                else:
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    self.lr_schedule[epoch] = self.opt.param_groups[0]["lr"]
                    if log_val_epoch:
                        file.close()

        # Apply best model params to model
        self.model.load_state_dict(self.best_model_params["state_dict"]) #self.model = 
        self.opt.load_state_dict(self.best_model_params["optimizer"]) #self.opt =
        print(f"Best model was found in epoch {self.best_model_params['epoch']}.")

        if log_val_epoch:
            file.close()

    def run_epoch(self, dataloader: DataLoader, device: str):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if self.model.training:
                self.opt.zero_grad()

            y_pred = self.model(x)

            loss = None
            loss = self.loss_func(y_pred, y)

            if self.model.training:
                loss.backward()
                self.opt.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def save_lr_schedule(self, path: str):
        """ save learning rate history to csv file"""
        with open(path, "w") as f:
            logging.info(f"Saving lr-schedule to {path}.")
            for epoch, lr in self.lr_schedule.items():
                f.write(f"{epoch},{lr}\n")

    def load_lr_schedule(self, path: pathlib.Path, case_2hp:bool=False):
        """ read lr-schedule from csv file"""
        # check if path contains lr-schedule, else use default one
        if not path.exists():
            logging.warning(f"Could not find lr-schedule at {path}. Using default lr-schedule instead.")
            path = pathlib.Path.cwd() / "Heat-Plume-Prediction_temp"/ "processing" / "lr_schedules"
            lr_schedule_file = "default_lr_schedule.csv" if not case_2hp else "default_lr_schedule_2hp.csv"
            path = path / lr_schedule_file

        with open(path, "r") as f:
            for line in f:
                epoch, lr = line.split(",")
                self.lr_schedule[int(epoch)] = float(lr)
