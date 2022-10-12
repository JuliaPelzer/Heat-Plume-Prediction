import pickle
import os
# from dataclasses import dataclass

# @dataclass
# class Location():
#     name:str
#     path:str=None

#     def check_path(self):
#         assert os.path.exists(self.path), f"Path {self.path} does not exist"
# TODO later


def save_pickle(data_dict, file_name):
    """Save given data dict to pickle file file_name in models/
    
    Parameters
    ----------
    data_dict : e.g. {"dataset": dataset,
        "cifar_mean": cifar_mean,
        "cifar_std": cifar_std}
    file_name : str
        Name of the file to be saved, ends with .p
    """
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(data_dict, open(os.path.join(directory, file_name), 'wb', 5))
