from data.dataset import DatasetSimulationData
from data.dataloader import DataLoader, _datapoint_to_tensor_including_channel
from data.transforms import NormalizeTransform, ComposeTransform, ReduceTo2DTransform, PowerOfTwoTransform, ToTensorTransform
from data.utils import PhysicalVariables, DataPoint
from networks.unet_leiterrl import weights_init
from tqdm.auto import tqdm
import logging

from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss
import torch.nn.functional as F
from torch import zeros, Tensor

from networks.unet_leiterrl import TurbNetG, UNet
from networks.dummy_network import DummyNet


def init_data(reduce_to_2D: bool = True, reduce_to_2D_wrong: bool = False, overfit: bool = False, normalize: bool = True, just_plotting: bool = False, batch_size: int = 100, inputs: str = "xyzpt",
              dataset_name: str = "approach2_dataset_generation_simplified/dataset_HDF5_testtest", path_to_datasets: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth"):
    """
    Initialize dataset and dataloader for training.

    Parameters
    ----------
    reduce_to_2D : If true, reduce the dataset to 2D instead of 3D
    overfit : If true, only use a small subset of the dataset for training, achieved by cheating: changing the split ratio
    normalize : If true, normalize the dataset, usually the case; not true for testing input data magnitudes etc 
    dataset_name : Name of the dataset to use (has to be the same as in the folder)
    batch_size : Size of the batch to use for training

    Returns
    -------
    datasets : dict
        Dictionary of datasets, with keys "train", "val", "test"
    dataloaders : dict
        Dictionary of dataloaders, with keys "train", "val", "test"
    """
    assert isinstance(reduce_to_2D, bool) and isinstance(reduce_to_2D_wrong, bool) and isinstance(overfit, bool) and isinstance(normalize, bool) and isinstance(
        just_plotting, bool), "input parameters reduce_to_2D, reduce_to_2D_wrong, overfit, normalize, just_plotting have to be bool"
    assert isinstance(
        batch_size, int), "input parameter batch_size has to be int"
    assert isinstance(dataset_name, str) and isinstance(
        path_to_datasets, str), "input parameters dataset_name, path_to_datasets have to be str"

    datasets = {}
    transforms_list = [
        ToTensorTransform(), PowerOfTwoTransform(oriented="left")]
    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform(
            reduce_to_2D_wrong=reduce_to_2D_wrong))
    if normalize:
        transforms_list.append(NormalizeTransform(reduced_to_2D=reduce_to_2D))
    logging.info(f"transforms_list: {transforms_list}")

    transforms = ComposeTransform(transforms_list)
    split = {'train': 0.6, 'val': 0.2, 'test': 0.2} if not overfit else {
        'train': 1, 'val': 0, 'test': 0}

    # just plotting (for Marius)
    if just_plotting:
        split = {'train': 1, 'val': 0, 'test': 0}
        transforms = None

    input_list = [inputs[i] for i in range(len(inputs))]
    for i in input_list:
        assert i in ["x", "y", "z", "p",
                     "t"], "input parameter inputs has to be a string of characters, each of which is either x, y, z, p, t"
    input_vars = []
    if 'x' in input_list:
        input_vars.append("Liquid X-Velocity [m_per_y]")
    if 'y' in input_list:
        input_vars.append("Liquid Y-Velocity [m_per_y]")
    if 'z' in input_list:
        input_vars.append("Liquid Z-Velocity [m_per_y]")
    if 'p' in input_list:
        input_vars.append("Liquid_Pressure [Pa]")
    if 't' in input_list:
        input_vars.append("Temperature [C]")
    # if '-' not in input_list:
    input_vars.append("Material_ID")

    for mode in ['train', 'val', 'test']:
        temp_dataset = DatasetSimulationData(
            dataset_name=dataset_name, dataset_path=path_to_datasets,
            transform=transforms, input_vars_names=input_vars,
            output_vars_names=["Temperature [C]", "Liquid X-Velocity [m_per_y]",
                               "Liquid Y-Velocity [m_per_y]", "Liquid Z-Velocity [m_per_y]"],  # . "Liquid_Pressure [Pa]"
            mode=mode, split=split
        )
        datasets[mode] = temp_dataset

    # Create a dataloader for each split.
    dataloaders = {}
    for mode in ['train', 'val', 'test']:
        temp_dataloader = DataLoader(
            dataset=datasets[mode],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        dataloaders[mode] = temp_dataloader
    print(f'init done [total number of datapoints/runs: {len(datasets["train"])+len(datasets["val"])+len(datasets["test"])}]')


    # TODO make dataset usable for tran_model: put properties into channel dimension
    return datasets, dataloaders


def train_model(model, dataloaders, loss_fn, n_epochs: int, lr: float, name_folder: str = "default", debugging: bool = False):
    """
    Train the model for a certain number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        Choose the model to train with varying structure etc
    dataloaders : dict
        Dictionary of dataloaders, with keys "train", "val", "test"
    loss_fn : torch.nn.Module
        Loss function to use, e.g. MSELoss() - depends on the model
    n_epochs : int
        Number of epochs to train for
    lr : float
        Learning rate to use

    Returns
    -------
        model : torch.nn.Module
            Trained model, ready to be applied to data;
            returns data in format of (Batch_id, channel_id, H, W)
    """

    # initialize Adam optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    # initialize tensorboard
    if not debugging:
        writer = SummaryWriter(f"runs/{name_folder}")
    loss_hist = []
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    model.apply(weights_init)
    epochs = tqdm(range(n_epochs), desc="epochs")
    for epoch in epochs:
        for batch_idx, data_values in enumerate(dataloaders["train"]):
            # dataloaders["train"] contains 3 data points
            x = data_values.inputs.float()
            y = data_values.labels.float()

            model.zero_grad()
            optimizer.zero_grad()

            y_out = model(x)  # dimensions: (batch-datapoint_id, channel, x, y, (z))
            mse_loss = loss_fn(y_out, y)
            loss = mse_loss

            loss.backward()
            optimizer.step()
            epochs.set_postfix_str(f"loss: {loss.item():.4f}")

            loss_hist.append(loss.item())
            scheduler.step(loss)

            if not debugging:
                writer.add_scalar("loss", loss.item(), epoch *
                                  len(dataloaders["train"])+batch_idx)
                writer.add_image("y_out_0", y_out[0, 0, :, :], dataformats="WH",
                                 global_step=epoch*len(dataloaders["train"])+batch_idx)
                #writer.add_image("y_out_1", y_out[0,1,:,:], dataformats="WH")

        if not debugging:
            writer.add_image("x_0", x[0, 0, :, :], dataformats="WH")
            # writer.add_image("x_1", x[0,1,:,:], dataformats="WH")
            # writer.add_image("x_2", x[0,2,:,:], dataformats="WH")
            # writer.add_image("x_3", x[0,3,:,:], dataformats="WH")
            # writer.add_image("x_4", x[0,4,:,:], dataformats="WH")
            # writer.add_image("y_0", y[0,0,:,:], dataformats="WH")
            # #writer.add_image("y_1", y[0,1,:,:], dataformats="WH")

    # if not debugging:
    #     writer.add_graph(model, x)
    print('Finished Training')

    return loss_hist


if __name__ == "__main__":
    
    # parameters of model and training
    loss_fn = MSELoss()
    n_epochs = 1 #60000
    lr=0.0004 #0.0004

    #model = TurbNetG(channelExponent=4, in_channels=4, out_channels=2)
    unet_model = UNet(in_channels=5, out_channels=1).float()
    # TODO too many in channels for unet?
    fc_model = DummyNet().float()
    # model.to(device)

    # init data
    datasets_2D, dataloaders_2D = init_data(dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_01", reduce_to_2D=True, overfit=True)
    # train model
    train_model(unet_model, dataloaders_2D, loss_fn, n_epochs, lr)

    # datasets, dataloaders = init_data(dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_10", reduce_to_2D=False, batch_size=4)

    # for dataloader in dataloaders.values():
    #     for _, datapoint in enumerate(dataloader):
    #         x = datapoint.inputs.float()
    #         y = datapoint.labels.float()