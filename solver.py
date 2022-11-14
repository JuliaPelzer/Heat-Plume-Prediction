## mainly copied from I2DL course at TUM, Munich
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import MSELoss, Module
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
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        # TODO include writer for tensorboard
        self.model:Module = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.loss_func = loss_func

        self.opt = optimizer(self.model.parameters(), learning_rate)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.opt)

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

    def _step(self, X, y, validation=False):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.

        :param X: batch of training features
        :param y: batch of corresponding training labels
        :param validation: Boolean indicating whether this is a training or
            validation step

        :return loss: Loss between the model prediction for X and the target
            labels y
        """
        loss = None

        # self.model.zero_grad() #
        # self.opt.zero_grad() #

        # Forward pass
        y_pred = self.model(X)
        # Compute loss
        loss = self.loss_func(y_pred, y)
        # Add the regularization
        # loss += sum(self.model.reg.values()) #

        # Perform gradient update (only in train mode)
        if not validation:
            # Compute gradients
            loss.backward() #
            # self.opt.backward(y_pred, y) #
            # Update weights
            self.opt.step()

        return loss, y_pred

    def train(self, n_epochs=100, patience=None, name_folder: str = "default"):
        """
        Run optimization to train the model.
        """
        # initialize tensorboard
        if self.debug_output: #
            writer = SummaryWriter(f"runs/{name_folder}")

        epochs = tqdm(range(n_epochs), desc="epochs")
        # Start an epoch
        for epoch in epochs:

            # Iterate over all training samples
            train_epoch_loss = 0.0

            for batch_idx, data_values in enumerate(self.train_dataloader):
                # Unpack data
                X = data_values.inputs.float()
                y = data_values.labels.float()

                # Update the model parameters.
                validate = epoch == 0
                train_loss, y_pred = self._step(X, y, validation=validate) #

                self.train_batch_loss.append(train_loss)
                train_epoch_loss += train_loss

                self.scheduler.step(train_loss) # here or one further out

            train_epoch_loss /= len(self.train_dataloader)

            # self.opt.lr *= self.lr_decay
            if self.debug_output:
                writer.add_scalar("train_loss", train_epoch_loss.item(), epoch *
                                  len(self.train_dataloader)+batch_idx)
                writer.add_image("y_out", y_pred[0, 0, :, :], dataformats="WH",
                                 global_step=epoch*len(self.train_dataloader)+batch_idx)

            # Iterate over all validation samples
            val_epoch_loss = 0.0

            for batch_idx, data_values in enumerate(self.val_dataloader):
                # Unpack data
                X = data_values.inputs.float()
                y = data_values.labels.float()

                # Compute Loss - no param update at validation time!
                val_loss, _ = self._step(X, y, validation=True) #
                self.val_batch_loss.append(val_loss)
                val_epoch_loss += val_loss

            val_epoch_loss /= len(self.val_dataloader)

            if self.debug_output:
                writer.add_scalar("val_loss", val_epoch_loss.item(), epoch *
                                  len(self.train_dataloader)+batch_idx)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_epoch_loss)
            self.val_loss_history.append(val_epoch_loss)

            if self.debug_output and epoch % self.print_every == 0:
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.4f}, val loss: {val_epoch_loss:.4f}")
                # print('(Epoch %d / %d) train loss: %f; val loss: %f' % (
                #     epoch + 1, epochs, train_epoch_loss, val_epoch_loss))

            # Keep track of the best model
            self.update_best_loss(val_epoch_loss, train_epoch_loss)
            if patience and self.current_patience >= patience:
                print("Stopping early at epoch {}!".format(epoch))
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