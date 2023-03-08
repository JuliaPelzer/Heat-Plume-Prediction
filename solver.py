## mainly copied from I2DL course at TUM, Munich
from torch.optim import Adam
from torch.nn import MSELoss, Module
from tqdm.auto import tqdm
from data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks.unet_leiterrl import weights_init


class Solver(object):
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_func=MSELoss(),
        learning_rate=1e-3,
        optimizer=Adam,
    ):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - debug_output: Boolean; if set to false then no output will be printed during
          training.
        """
        self.model: Module = model
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opt = optimizer(self.model.parameters(), learning_rate)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self._reset()

    def train(
        self,
        device,
        n_epochs: int = 100,
        name_folder: str = "default",
    ):
        """
        Run optimization to train the model.
        """
        # initialize tensorboard
        writer = SummaryWriter(f"runs/{name_folder}")
        epochs = tqdm(range(n_epochs), desc="epochs",disable=True)


        self.model = self.model.to(device)
        for epoch in epochs:
            try:
                # Training
                self.model.train()
                train_epoch_loss = self.run_epoch(self.train_dataloader, device)

                # Validation
                self.model.eval()
                val_epoch_loss = self.run_epoch(self.val_dataloader, device)

                # Logging
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                epochs.set_postfix_str(
                    f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")

            except KeyboardInterrupt:
                try:
                    new_lr = float(input("\nNew learning rate: "))
                except ValueError as e:
                    print(e)
                else:
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr

    def run_epoch(self,dataloader: DataLoader, device):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            loss = self._step(x, y)
            epoch_loss += loss
        epoch_loss /= len(dataloader)
        return epoch_loss


    def _reset(self):
        """
        Don't call this manually.
        """
        self.model.apply(weights_init)

    def _step(self, x, y):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        loss = None
        if self.model.training:
            self.opt.zero_grad()

        y_pred = self.model(x)
        loss = self.loss_func(y_pred, y)

        if self.model.training:
            loss.backward()
            self.opt.step()

        return loss.detach().item()
