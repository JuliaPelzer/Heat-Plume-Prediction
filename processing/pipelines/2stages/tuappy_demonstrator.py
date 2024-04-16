import argparse
import logging
import os
import pathlib
import sys
import time

import yaml

# sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN") # relevant for remote
# sys.path.append("/home/pelzerja/Development/1HP_NN")  

from preprocessing.domain_classes.domain import Domain
from preprocessing.prepare_2ndstage import (load_and_prepare_for_2nd_stage,
                                            prepare_dataset_for_2nd_stage)
from preprocessing.prepare_paths import Paths2HP, set_paths_2hpnn
from processing.networks.unet import UNet


def demonstrator(dataset_large_name: str, preparation_case: str, model_name_2HP: str = None, device: str = "cuda:0", destination_demo: str = "", visualize: bool = False):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    - expects the input to have just one datapoint
    """
    run_id = 0 # Achtung only used for loading
    paths: Paths2HP
    paths, inputs_1hp, models_2hp_dir = set_paths_2hpnn(dataset_large_name, preparation_case)
    model_2hp_path = models_2hp_dir / model_name_2HP
    destination_demo = pathlib.Path("runs/domain/default") if destination_demo == "" else pathlib.Path("runs/domain")/destination_demo
    destination_demo.mkdir(exist_ok=True, parents=True)

    # prepare domain and boxes with 1hpnn applied
    if not os.path.exists(paths.dataset_1st_prep_path) or not os.path.exists(paths.datasets_boxes_prep_path):
        domain, single_hps = prepare_dataset_for_2nd_stage(paths, inputs_1hp, device)
    else:
        domain, single_hps = load_and_prepare_for_2nd_stage(paths, inputs_1hp, run_id, device)
    print(f"Dataset prepared ({paths.datasets_boxes_prep_path}) - with boxes of primary and other temp field and label")

    # load 2ndstage model
    model_2HP = UNet(in_channels=2).float()
    model_2HP.load(model_2hp_path, device)
    model_2HP.eval()

    # apply 2ndstage model
    for hp in single_hps:
        if not hp.other_temp_field.all() == 10.6:
            hp.output = hp.apply_nn(model_2HP, inputs="interim_outputs")

            if visualize:
                hp.plot_and_reverse_norm(domain, dir=destination_demo)
            domain.add_hp(hp, hp.output)
        else:
            # if no overlap: dont apply 2HP-NN and just use 1HP-NN
            domain.add_hp(hp, hp.primary_temp_field)
            print("I didnt expect this to happen")

    if visualize:
        domain.plot("t", folder=destination_demo, name="domain", format_fig="png")
    domain.save(destination_demo, run_id)

    with open(destination_demo / "args.yaml", "w") as file:
        yaml.safe_dump({"dataset_large_name": dataset_large_name, "preparation_case": preparation_case, "model_name_2HP": model_name_2HP, "device": device, }, file, )

    measure = True
    if measure:
        measure_accuracy(domain, single_hps, destination_demo, visualize)

def measure_accuracy(domain: Domain, single_hps: list, destination_demo: str, visualize: bool = False):
    avg_loss_mae = 0
    avg_loss_rmse = 0
    avg_mae = {0: 0, 1: 0}
    avg_mse = {0: 0, 1: 0}
    num_split = {0: 0, 1: 0}

    for id_hp, hp in enumerate(single_hps):
        # measurements in normed mode
        loss_mae, loss_mse = hp.measure_accuracy(domain, plot_args=[visualize, destination_demo / f"plot_error_hp{hp.id}.png"])
        avg_mae[id_hp] += loss_mae
        avg_mse[id_hp] += loss_mse
        num_split[id_hp] += 1
        avg_loss_mae += loss_mae
        avg_loss_rmse += loss_mse

    avg_loss_mae /= len(single_hps)
    avg_loss_rmse /= len(single_hps)
    for id_hp in range(len(single_hps)):
        avg_mae[id_hp] /= num_split[id_hp]
        avg_mse[id_hp] /= num_split[id_hp]

    with open(destination_demo / "measurements_apply.yaml", "w") as file:
        yaml.safe_dump({
                "timestamp_end": time.ctime(),
                "avg_loss_mae": float(avg_loss_mae),
                "avg_loss_mse": float(avg_loss_rmse),
                "avg_mae": {int(k): float(v) for k, v in avg_mae.items()},
                "avg_mse": {int(k): float(v) for k, v in avg_mse.items()},
                "num_split": {int(k): int(v) for k, v in num_split.items()},
            },file,)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation_case", type=str, default="gksi100")
    parser.add_argument("--dataset_large", type=str, default="dataset_2d_small_10dp")
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--model_2hp", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--visualize", type=bool, default=False)
    args = parser.parse_args()

    demonstrator(
        dataset_large_name=args.dataset_large,
        preparation_case=args.preparation_case,
        model_name_2HP=args.model_2hp,
        device=args.device,
        destination_demo=args.destination,
        visualize=args.visualize,
    )
