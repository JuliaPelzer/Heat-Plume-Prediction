from typing import Dict
import csv
import pandas as pd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def append_results_to_csv(results:Dict, name_file_destination:str):
    """Save results to csv file"""
    with open(name_file_destination, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writerow(results)


def init_results_csv_file(name_file_destination:str):
    """Initialize csv file for results"""
    columns = ["timestamp", "model", "dataset", "overfit", "inputs", "n_epochs", "lr", "error_mean", "error_max", "duration", "name_destination_folder"]
    # df = pd.DataFrame(columns=columns)
    # df.to_csv(name_file_destination, index=False)
    with open(name_file_destination, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()