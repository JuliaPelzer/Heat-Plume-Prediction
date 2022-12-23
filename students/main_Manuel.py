from data.dataset_loading import init_data
from solver import Solver
from networks.dummy_network import DummyNet
from visualization.visualize_data import plot_sample
from torch.nn import MSELoss
from torch import cuda, device
from utils.utils_networks import count_parameters
import datetime as dt
import sys
import logging
import numpy as np

dataset_name = "approach2_dataset_generation_simplified/dataset_HDF5_testtest"
path_to_datasets = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets"

def run_experiment(n_epochs:int=1000, lr:float=5e-3, inputs:str="pk", model_choice="unet", name_folder_destination:str="default", dataset_name:str="small_dataset_test", overfit=True):
    
    time_begin = dt.datetime.now()
    
    # parameters of model and training
    loss_fn = MSELoss()
    n_epochs = n_epochs
    lr=float(lr)
    reduce_to_2D=True
    reduce_to_2D_xy=True
    overfit=overfit

    # init data
    datasets_2D, dataloaders_2D = init_data(dataset_name=dataset_name,  
        reduce_to_2D=reduce_to_2D, reduce_to_2D_xy=reduce_to_2D_xy,
        inputs=inputs, labels="t", overfit=overfit)

    # model choice
    in_channels = len(inputs)+1
    if model_choice == "fc":
        size_domain_2D = datasets_2D["train"].dimensions_of_datapoint
        if reduce_to_2D:
            # TODO order here or in dummy_network(size) messed up
            size_domain_2D = size_domain_2D[1:]
        # transform to PowerOfTwo
        size_domain_2D = [2 ** int(np.log2(dimension)) for dimension in size_domain_2D]
        
        model = DummyNet(in_channels=in_channels, out_channels=1, size=size_domain_2D).float()
    else:
        print("model choice not recognized")
        sys.exit()

    device_used = device('cuda' if cuda.is_available() else 'cpu')
    if not device_used == 'cuda':
        logging.info(f"Using {device_used} device")
    model.to(device_used)

    number_parameter = count_parameters(model)
    logging.info(f"Model {model_choice} with number of parameters: {number_parameter}")

    # train model
    if overfit:
        solver = Solver(model, dataloaders_2D["train"], dataloaders_2D["train"], 
                    learning_rate=lr, loss_func=loss_fn)
    else:
        solver = Solver(model, dataloaders_2D["train"], dataloaders_2D["val"], 
                    learning_rate=lr, loss_func=loss_fn)
    patience_for_early_stopping = 50
    solver.train(device_used, n_epochs=n_epochs, name_folder=name_folder_destination, patience=patience_for_early_stopping)

    # visualization
    if overfit:
        error, error_mean, final_max_error = plot_sample(model, dataloaders_2D["train"], device_used, name_folder_destination, plot_name="plot_learned_test_sample")
    else:
        error, error_mean, final_max_error = plot_sample(model, dataloaders_2D["test"], device_used, name_folder_destination, plot_name="plot_learned_test_sample", plot_one_bool=False)
    
    # save model - TODO : both options currently not working
    # @Manuel: Wenn du mich dran erinnerst, schicke ich dir hierzu ein Update, wenn es läuft
    # save(model, str(name_folder)+str(dataset_name)+str(inputs)+".pt")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL

    cla = sys.argv
    kwargs = {}
    if len(cla) >= 2:
        kwargs["n_epochs"] = int(cla[1])
        if len(cla) >= 3:
            kwargs["lr"] = float(cla[2])
            if len(cla) >= 4:
                kwargs["model_choice"] = cla[3]
                if len(cla) >= 5:
                    kwargs["inputs"] = cla[4]
                    if len(cla) >= 6:
                        kwargs["name_folder_destination"] = cla[5]
                        if len(cla) >= 7:
                            kwargs["dataset_name"] = cla[6]

    run_experiment(**kwargs)