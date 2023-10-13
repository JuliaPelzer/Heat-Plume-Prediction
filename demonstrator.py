import argparse
import logging
import os
import pathlib
import sys
import time

import yaml
from tqdm.auto import tqdm

from domain_classes.domain import Domain
from domain_classes.heat_pump import HeatPump
from domain_classes.utils_2hp import check_all_datasets_prepared, set_paths_2hpnn
from data_stuff.utils import SettingsTraining, load_yaml
from networks.unet import UNet
from utils.prepare_paths import Paths2HP, set_paths_2hpnn
from utils.utils import beep
from preprocessing.prepare_2ndstage import prepare_dataset_for_2nd_stage



def demonstrator(dataset_large_name: str, preparation_case: str, model_name_2HP: str = None, device: str = "cuda:0", stages: int = 2, destination_dir: str = "", visualize: bool = False):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    - expects the input to have just one datapoint
    """
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

    paths: Paths2HP
    paths, inputs_1hp, dataset_2hp_prep = set_paths_2hpnn(dataset_large_name, preparation_case, model_name_2HP)

    if not os.path.exists(paths.datasets_boxes_prep_path):
        domain, single_hps, info = prepare_dataset_for_2nd_stage(paths, dataset_large_name, inputs_1hp, device)
    # else:
    #     domain, single_hps, info = load_todo(paths.datasets_boxes_prep_path)
    print(f"Dataset prepared ({paths.datasets_boxes_prep_path})")

    # load 2ndstage model
    model = UNet(in_channels=2).float()
    model.load(paths.destination_dir, device)
    model.to(device)
    model.eval()

    # # TODO or do I only want dataset_1st_prep_path?

    # destination_2hpnn = pathlib.Path(os.getcwd(), "runs", domain_prepared_dir.name)
    # destination_2hpnn.mkdir(exist_ok=True)

    # # list_runs = os.listdir(os.path.join(domain_prepared_dir, "Inputs"))
    # # for run_file in tqdm(list_runs, desc="Apply 2HP-NN", total=len(list_runs)):
    #     # domain = Domain(domain_prepared_dir, stitching_method="max", file_name=run_file)

    #     # apply 2HP-NN
    #     time_apply_2nn = time.perf_counter()
    #     for hp in single_hps:
    #         if not hp.other_temp_field.all() == 10.6:
    #             hp.output = hp.apply_nn(model_2HP, inputs="interim_outputs")
    #             hp.output = domain.reverse_norm(hp.output, property="Temperature [C]")

    #             if False:
    #                 hp.label = domain.reverse_norm(hp.label, property="Temperature [C]")
    #                 hp.save_pred(run_id, pathlib.Path(os.getcwd(), "runs", domain_prepared_dir.name), inputs_all=[hp.output])

    #             domain.add_hp(hp, hp.output)
    #         else:
    #             # if no overlap: dont apply 2HP-NN and just use 1HP-NN
    #             domain.add_hp(hp, hp.primary_temp_field)
    #             print("I didnt expect this to happen")
    #     avg_time_apply_2nn += time.perf_counter() - time_apply_2nn

    #     time_measure_and_visualize = time.perf_counter()
    #     for id_hp, hp in enumerate(single_hps):
    #         loss_mae, loss_mse = hp.measure_accuracy(domain, plot_args=[False, destination_2hpnn / f"plot_hp{num_hps_overall}.png"])
    #         avg_mae[id_hp] += loss_mae
    #         avg_mse[id_hp] += loss_mse
    #         num_split[id_hp] += 1
    #         avg_loss_mae += loss_mae
    #         avg_loss_mse += loss_mse
    #         num_hps_overall += 1
    #     avg_time_measure_and_visualize += time.perf_counter() - time_measure_and_visualize
    #     domain.plot("t", folder=destination_2hpnn, name=run_id)
    #     num_dp_valid += 1
    #     # domain.save(destination_2hp_nn, run_id)
    
    # # avg measurements
    # avg_time_prep_1hp /= num_dp_valid
    # avg_time_apply_1nn /= num_dp_valid
    # avg_time_prep_2hp /= num_hps_overall
    # avg_time_apply_2nn /= num_hps_overall
    # avg_time_measure_and_visualize /= num_hps_overall
    # avg_loss_mae /= num_hps_overall
    # avg_loss_mse /= num_hps_overall
    # for id_hp in avg_mae.keys():
    #     avg_mae[id_hp] /= num_split[id_hp]
    #     avg_mse[id_hp] /= num_split[id_hp]
    #     # TODO after reasonable training: check, if still avg_x[0] so different to avg_x[1]
    #     # if not: remove the whole part about avg_mae and avg_mse and num_split

    # with open(destination_2hpnn / "measurements_apply.yaml", "w") as file:
    #     yaml.safe_dump(
    #         {
    #             "time whole process in sec": time.perf_counter() - time_begin,
    #             "timestamp_begin": timestamp_begin,
    #             "timestamp_end": time.ctime(),
    #             "avg_time_prep_1hp in sec (per dp)": avg_time_prep_1hp,
    #             "avg_time_apply_1nn (incl. renorming) in sec (per dp)": avg_time_apply_1nn,
    #             "avg_time_prep_2hp (incl. norming) in sec (per hp)": avg_time_prep_2hp,
    #             "avg_time_apply_2nn (incl. renorming) in sec (per hp)": avg_time_apply_2nn,
    #             "avg_time_measure_and_visualize in sec (per hp)": avg_time_measure_and_visualize,
    #             "time to prepare paths etc. in sec": time_prepare,
    #             "number valid datapoints": num_dp_valid,
    #             "avg_loss_mae": float(avg_loss_mae),
    #             "avg_loss_mse": float(avg_loss_mse),
    #             "number of heat pump boxes in training": num_hps_overall,
    #             "avg_mae": {int(k): float(v) for k, v in avg_mae.items()},
    #             "avg_mse": {int(k): float(v) for k, v in avg_mse.items()},
    #             "num_split": {int(k): int(v) for k, v in num_split.items()},
    #         },
    #         file,
    #     )
    
    # with open(destination_2hpnn / "args.yaml", "w") as file:
    #     yaml.safe_dump(
    #         {
    #             "dataset_large_name": dataset_large_name,
    #             "preparation_case": preparation_case,
    #             "model_name_2HP": model_name_2HP,
    #             "device": device,
    #         },
    #         file,
    #     )

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation_case", type=str, default="gksi100")
    parser.add_argument("--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm") # TODO defaults
    parser.add_argument("--destination_dir", type=str, default="")
    parser.add_argument("--model_2hp", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--visualize", type=bool, default=False)
    args = parser.parse_args()

    demonstrator(
        dataset_large_name=args.dataset_large,
        preparation_case=args.preparation_case,
        model_name_2HP=args.model_2hp,
        device=args.device,
        stages=2,
        destination_dir=args.destination_dir,
        visualize=args.visualize,
    )
