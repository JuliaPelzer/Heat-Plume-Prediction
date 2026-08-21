import logging
from pathlib import Path
import time
from dataclasses import dataclass
import optuna
from torch import manual_seed, no_grad
from torch.nn import Module, modules, MSELoss, HuberLoss
from torch.optim import Adam, Optimizer, LBFGS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from copy import deepcopy

from postprocessing.visualization import visualizations, interim_visu
from processing.networks.model import weights_init
from utils.utils_args import save_yaml

@dataclass
class Solver(object):
    model: Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    loss_func: modules.loss._Loss = MSELoss()
    learning_rate: float = 1e-4
    opt: Optimizer = Adam
    finetune: bool = False
    best_model_params: dict = None
    metrics: dict = None
    val_interval: int = 5 #TODO 50

    def __post_init__(self):
        self.opt = self.opt(self.model.parameters(),self.learning_rate, weight_decay=0.0) # TODO redo to 1e-4 ?
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}

        if not self.finetune:
            self.model.apply(weights_init)
        
        self.metrics: dict = {"Huber": HuberLoss(), }

    def train(self, args: dict, optuna_trial=None):
        manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        if optuna_trial:
            log_dir = args["destination"] / f"trial{optuna_trial.number}"
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        else:
            log_dir = args["destination"]
        writer = SummaryWriter(log_dir=log_dir)
        device = args["device"]

        epochs = tqdm(range(args["epochs"]), desc="epochs", disable=False)
        for epoch in epochs:
            try:
                # Set lr according to schedule
                if epoch in self.lr_schedule.keys():
                    self.opt.param_groups[0]["lr"] = self.lr_schedule[epoch]
                    
                # Training
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device, eval_bool=False)
                
                # Validation
                if epoch % self.val_interval == 0 or epoch == args["epochs"] - 1:
                    with no_grad():
                        self.model.eval()
                        val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device, eval_bool=True)
                        val_epoch_loss = other_losses_val["Huber"]

                        # Keep best model
                        if self.best_model_params is None or val_epoch_loss < self.best_model_params["loss"]:
                            self.best_model_params = {
                                "epoch": epoch,
                                "loss": val_epoch_loss,
                                "train loss": train_epoch_loss,
                                "state_dict": deepcopy(self.model.state_dict()),
                                "optimizer": deepcopy(self.opt.state_dict()),
                                "training time in sec": (time.perf_counter() - start_time),
                            }

                # # Logging
                for metric_name, metric_value in other_losses_val.items():
                    writer.add_scalar(f"val {metric_name}", metric_value, epoch)
                for metric_name, metric_value in other_losses_train.items():
                        writer.add_scalar(f"train {metric_name}", metric_value, epoch)

                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")    

                if optuna_trial:
                    optuna_trial.report(val_epoch_loss, epoch)
                    if optuna_trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            except KeyboardInterrupt:
                # allows to interrupt training with ctrl+c to change the lr manually
                try:
                    model_tmp = deepcopy(self.model)
                    model_tmp.load_state_dict(self.best_model_params["state_dict"])
                    model_tmp.to(args["device"])
                    model_tmp.save(args["destination"], model_name=f"interim_model_e{epoch}.pt")
                    interim_visu(model_tmp, self.val_dataloader, path_desti=args["destination"] / f"interim_visu_e{epoch}.png", device=args["device"])
                except Exception as e:
                    logging.error(e)
                    
                try:
                    new_lr = float(input("\nNew learning rate: "))
                except ValueError as e:
                    print(e)
                else:
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    self.lr_schedule[epoch] = self.opt.param_groups[0]["lr"]

        # Apply best model params to model
        self.model.load_state_dict(self.best_model_params["state_dict"]) 
        self.opt.load_state_dict(self.best_model_params["optimizer"])
        print(f"Best model was found in epoch {self.best_model_params['epoch']}.")

        return self.best_model_params["loss"]

    def run_epoch(self, dataloader: DataLoader, device: str, eval_bool: bool = False):
        epoch_loss = 0.0

        metric_totals = {metric_name: 0.0 for metric_name in self.metrics}
        max_abs_error = 0.0
        peak_abs_error = 0.0
        metric_values = {}

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if self.model.training:
                self.opt.zero_grad()

            y_pred = self.model(x)
            if self.model.__name__() == "UNetNoPad2":
                required_size = y_pred.shape[2:]
                start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
                y = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

            loss = self.loss_func(y_pred, y)

            if eval_bool:
                abs_error = (y_pred.detach() - y.detach()).abs()
                max_abs_error = max(max_abs_error, abs_error.max().item())
                peak_abs_error = max(peak_abs_error, self.error_at_label_peak(abs_error, y.detach()))
                for metric_name, metric in self.metrics.items():
                    metric_totals[metric_name] += metric(y_pred, y).detach().item()

            if self.model.training:
                loss.backward()
                self.opt.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader)

        # Calculate metrics
        if eval_bool:
            metric_values = {
                metric_name: metric_total / len(dataloader)
                for metric_name, metric_total in metric_totals.items()
            }
            metric_values["max abs error"] = max_abs_error
            metric_values["peak abs error"] = peak_abs_error

        return epoch_loss, metric_values

    @staticmethod
    def error_at_label_peak(abs_error, label):
        label_flat = label.flatten(start_dim=2)
        error_flat = abs_error.flatten(start_dim=2)
        peak_indices = label_flat.argmax(dim=2, keepdim=True)
        return error_flat.gather(2, peak_indices).max().item()

    def save_lr_schedule(self, path: str):
        """ save learning rate history to csv file"""
        with open(path, "w") as f:
            logging.info(f"Saving lr-schedule to {path}.")
            for epoch, lr in self.lr_schedule.items():
                f.write(f"{epoch},{lr}\n")

    def load_lr_schedule(self, path: Path):
        """ read lr-schedule from csv file"""
        # check if path contains lr-schedule, else use default one
        if not path.exists():
            logging.warning(f"Could not find lr-schedule at {path}. Using default lr-schedule instead.")
            path = Path.cwd() / "processing" / "default_lr_schedule.csv"

        with open(path, "r") as f:
            for line in f:
                epoch, lr = line.split(",")
                self.lr_schedule[int(epoch)] = float(lr)

    def save_metrics_separate_yaml(self, destination: Path, no_params:int, max_epochs:int, training_time:float, device: str = "cpu"):
        # prepare data as dict

        self.model.eval()
        train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device, eval_bool=True) 
        val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device, eval_bool=True)

        metrics = {}
        metrics["no_params"] = no_params
        metrics["max_epochs"] = max_epochs
        metrics["training_time [s]"] = training_time
        metrics["best_epoch"] = self.best_model_params["epoch"]
        metrics["train"] = other_losses_train
        metrics["train"]["train loss"] = train_epoch_loss
        metrics["val"] = other_losses_val
        metrics["val"]["val loss"] = val_epoch_loss

        # save data as yaml
        save_yaml(metrics, destination / "measurements.yaml")