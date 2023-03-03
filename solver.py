## mainly copied from I2DL course at TUM, Munich
from torch.optim import Adam, lr_scheduler
from torch.nn import MSELoss, Module
import torch
from tqdm.auto import tqdm
from data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from networks.unet_leiterrl import weights_init

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    or regression models.
    The Solver performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, learning_rate to the constructor.
    You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists
    containing the losses of the model on the training and validation set at
    each epoch.
    """

    def __init__(self, model, train_dataloader:DataLoader, val_dataloader:DataLoader,
                 loss_func=MSELoss(), learning_rate=1e-3,
                 optimizer=Adam, debug_output=True, print_every=1, lr_decay=1.0,
                 **kwargs):
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
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model:Module = model
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.opt = optimizer(self.model.parameters(), learning_rate)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.opt, factor = 0.5, cooldown = 10, verbose = True)

        self.debug_output = debug_output
        self.print_every = print_every

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.current_patience = 0
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.current_patience = 0
        self.model.apply(weights_init) #

    def _step(self, X, y, device, validation=False):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        :return loss: Loss between the model prediction for X and the target
            labels y
        """
        loss = None
        if not validation:
            self.opt.zero_grad()

        y_pred = self.model(X).to(device)  # Forward pass
        loss = self.loss_func(y_pred, y)    # Compute loss
        # loss += sum(self.model.reg.values())  # Add the regularization    #   
        if not validation:  # Perform gradient update (only in train mode)
            loss.backward() # Compute gradients     #
            # self.opt.backward(y_pred, y) #
            self.opt.step() # Update weights

        loss = loss.detach().item()
        y_pred = y_pred.detach()

        return loss, y_pred

    def train(self, device, n_epochs:int=100, patience:int=None, name_folder: str = "default"):
        """
        Run optimization to train the model.
        """
        # initialize tensorboard
        if self.debug_output:
            writer = SummaryWriter(f"runs/{name_folder}")

        # epochs = range(n_epochs)
        epochs = tqdm(range(n_epochs), desc="epochs")
        # Start an epoch
        for epoch in epochs:

            # Iterate over all training samples
            train_epoch_loss = 0.0
            for batch_idx, data_values in enumerate(self.train_dataloader):
                # Unpack data
                X = data_values.inputs.float().to(device)
                y = data_values.labels.float().to(device)

                # Train step + update model parameters
                validate = epoch == 0
                train_loss, y_pred = self._step(X, y, device, validation=validate)
                # self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss
            # train_epoch_loss /= (batch_idx+1)
            train_epoch_loss /= len(self.train_dataloader.dataset)
            train_max_error, train_pos_of_max_error = get_max_error_and_pos(y,y_pred)
            # Iterate over all validation samples
            val_epoch_loss = 0.0
            for batch_idx, data_values in enumerate(self.val_dataloader):
                # Unpack data
                X = data_values.inputs.float().to(device)
                y = data_values.labels.float().to(device)
                # Compute Loss - no param update at validation time!
                val_loss, y_pred_val = self._step(X, y, device, validation=True) #
                # self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss
            # val_epoch_loss /= (batch_idx+1)
            val_epoch_loss /= len(self.val_dataloader.dataset)
            self.scheduler.step(val_epoch_loss) #TODO test

            if self.debug_output:
                writer.add_scalar("train_loss", train_epoch_loss, epoch) # * len(self.train_dataloader.dataset)+batch_idx)
                writer.add_image("train_y_out", y_pred[0, 0, :, :], dataformats="WH", global_step=epoch) # *len(self.train_dataloader.dataset)+batch_idx)
                writer.add_image("val_y_out", y_pred_val[0, 0, :, :], dataformats="WH", global_step=epoch) # *len(self.train_dataloader.dataset)+batch_idx)
                writer.add_scalar("val_loss", val_epoch_loss, epoch) # * len(self.train_dataloader.dataset)+batch_idx)
            
            val_max_error, val_pos_of_max_error = get_max_error_and_pos(y,y_pred_val)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.debug_output and epoch % self.print_every == 0:
                epochs.set_postfix_str(f"train_max_loss {train_max_error:.2e} at {train_pos_of_max_error}, val_max_loss {val_max_error:.2e} at {val_pos_of_max_error} train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss)
            if patience and self.current_patience >= patience:
                print("Stopping early at epoch {}!".format(epoch))
                n_epochs = epoch
                break

        # At the end of training swap the best params into the model
        self.model.params = self.best_params

    def update_best_loss(self, val_loss, train_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"val_loss": val_loss, "train_loss": train_loss}
            self.best_params = self.model.parameters()
            self.current_patience = 0
        else:
            self.current_patience += 1

def get_max_error_and_pos(y,y_pred):
    max_value = torch.max(abs(y-y_pred))
    pos = find_pos(max_value, abs(y-y_pred))
    return max_value, pos

def find_pos(value, field):
    condition = field == value
    return condition.nonzero()