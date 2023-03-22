import os
import logging
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn import MSELoss, Module
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import DataLoader
from networks.unet import weights_init


class Solver(object):
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_func=MSELoss(),
        learning_rate=1e-5,
        optimizer=Adam,
    ):
        self.model: Module = model
        self.loss_func = loss_func
        self.opt = optimizer(self.model.parameters(), learning_rate, weight_decay=1e-4)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self._reset()

    def train(self, device: str, n_epochs: int = 100, name_folder: str = "default"):

        # initialize tensorboard
        writer = SummaryWriter(f"runs/{name_folder}")
        epochs = tqdm(range(n_epochs), desc="epochs",disable=False)

        self.model = self.model.to(device)
        for epoch in epochs:
            try:
                # Set lr according to schedule
                if epoch in self.lr_schedule.keys():
                    self.opt.param_groups[0]["lr"] = self.lr_schedule[epoch]
                    
                # Training
                self.model.train()
                train_epoch_loss = self.run_epoch(self.train_dataloader, device)

                # Validation
                self.model.eval()
                val_epoch_loss = self.run_epoch(self.val_dataloader, device)

                # Logging
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")

            except KeyboardInterrupt:
                try:
                    new_lr = float(input("\nNew learning rate: "))
                except ValueError as e:
                    print(e)
                else:
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    self.lr_schedule[epoch] = self.opt.param_groups[0]["lr"]

    def run_epoch(self, dataloader:DataLoader, device:str):
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

    def _reset(self):
        """
        Don't call this manually.
        """
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]} # contains the epoch and learning rate, when lr changes
        self.model.apply(weights_init)

    def save_lr_schedule(self, path:str):
        """ save learning rate history to csv file"""
        with open(path, "w") as f:
            print(f"Saving lr-schedule to {path}.")
            for epoch, lr in self.lr_schedule.items():
                f.write(f"{epoch},{lr}\n")
                        
    def load_lr_schedule(self, path:str):
        """ read lr-schedule from csv file"""
        # check if path contains lr-schedule, else use default one
        if not os.path.exists(path):
            logging.warning(f"Could not find lr-schedule at {path}. Using default lr-schedule instead.")
            path = os.path.join(os.getcwd(),"default_lr_schedule.csv")

        with open(path, "r") as f:
            for line in f:
                epoch, lr = line.split(",")
                self.lr_schedule[int(epoch)] = float(lr)