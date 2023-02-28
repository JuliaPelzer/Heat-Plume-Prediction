import logging
import numpy as np
from main import run_experiment
from data.utils import load_settings, save_settings

if __name__ == "__main__":
    # print eps
    print(f"Maximum achievable precision: for double precision: {np.finfo(np.float64).eps}, for single precision: {np.finfo(np.complex64).eps}")

    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    kwargs = load_settings(".", "settings_training")
    kwargs["path_to_datasets"] = "/home/pelzerja/pelzerja/test_nn/datasets"

    
    kwargs["lr"]=1e-4#7
    kwargs["overfit"] = True
    kwargs["dataset_name"] = "test_dataset_10"
    input_combis = ["p"] #, "t", "xy"] # "px", "py",
    kwargs["n_epochs"] = 20000
    for model in ["unet", "fc"]:
        kwargs["model_choice"] = model
        for input in input_combis:
            kwargs["inputs"] = input
            kwargs["name_folder_destination"] = f"{kwargs['model_choice']}_overfit_{kwargs['overfit']}_epochs_{kwargs['n_epochs']}_inputs_{kwargs['inputs']}_perm_iso_100dp"
            run_experiment(**kwargs)
    
    #tensorboard --logdir runs/