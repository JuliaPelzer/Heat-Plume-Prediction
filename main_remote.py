import logging
import numpy as np
from main import run_experiment
from data.utils import load_settings, save_settings
import sys

if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    cla = sys.argv
    kwargs = {}
    kwargs = load_settings(".", "settings_training")

    kwargs["dataset_name"] = "small_dataset_test"
    kwargs["path_to_datasets"] = "/home/pelzerja/pelzerja/test_nn/dataset_generation/datasets"
    kwargs["n_epochs"] = 1

    # print eps
    print(f"Maximum achievable precision: for double precision: {np.finfo(np.float64).eps}, for single precision: {np.finfo(np.complex64).eps}")
    
    run_experiment(**kwargs)
