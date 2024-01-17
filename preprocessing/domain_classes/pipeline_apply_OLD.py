import argparse
import logging
import os
import pathlib
import sys
import time

import yaml
from tqdm.auto import tqdm

from domain_classes.domain import Domain
from domain_classes.heat_pump import HeatPumpBox
from domain_classes.utils_2hp import check_all_datasets_prepared, set_paths_2hpnn
from utils.utils_data import SettingsPrepare, load_yaml
from processing.networks.unet import UNet
from preprocessing.prepare import prepare_dataset
from utils.utils import beep


def pipeline_apply_2HPNN(
    dataset_large_name: str,
    preparation_case: str,
    model_name_2HP: str = None,
    device: str = "cuda:0",
):
    # TODO clean up norming and reverse norming!
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

    # prepare large dataset if not done yet
    (datasets_raw_domain_dir, datasets_prepared_domain_dir, domain_prepared_dir, datasets_model_trained_with_path, model_1hp_path, model_2hp_path, _, inputs_prep
     ) = set_paths_2hpnn(dataset_large_name, preparation_case, model_name_2hp=model_name_2HP,)
    
    if not os.path.exists(domain_prepared_dir):
        args = SettingsPrepare(
            raw_dir=datasets_raw_domain_dir,
            datasets_dir=datasets_prepared_domain_dir,
            dataset_name=dataset_large_name,
            inputs_prep=inputs_prep,
            # name_extension=name_extension,
        )
        prepare_dataset(args,
            power2trafo=False,
            info=load_yaml(datasets_model_trained_with_path, "info"),
        )  # norm with data from dataset that NN was trained with!
        print(f"Domain {domain_prepared_dir} prepared")
    else:
        print(f"Domain {domain_prepared_dir} already prepared")

    destination_2hp_nn = pathlib.Path(os.getcwd(), "runs", domain_prepared_dir.name)
    destination_2hp_nn.mkdir(exist_ok=True)

    # load models
    model_1HP = UNet(in_channels = len(inputs_prep))
    model_1HP.load(model_1hp_path, device)
    model_2HP = UNet(in_channels = 2)
    model_2HP.load(model_2hp_path, device)
    
    num_dp_valid = 0
    avg_time_prep_1hp = 0
    avg_time_apply_1nn = 0
    avg_time_prep_2hp = 0
    avg_time_apply_2nn = 0
    avg_time_measure_and_visualize = 0
    num_hps_overall = 0
    avg_loss_mae = 0
    avg_loss_mse = 0
    avg_mae = {0: 0, 1: 0}
    avg_mse = {0: 0, 1: 0}
    num_split = {0: 0, 1: 0}

    time_prepare = time.perf_counter() - time_begin

    # apply 1HP-NN and 2HP-NN
    list_runs = os.listdir(os.path.join(domain_prepared_dir, "Inputs"))
    for run_file in tqdm(list_runs, desc="Apply 2HP-NN", total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        time_prep_1hp = time.perf_counter()
        domain = Domain(domain_prepared_dir, stitching_method="max", file_name=run_file)
        # generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        # generate 1hp-boxes and extract information like perm and ids etc.
        single_hps = domain.extract_hp_boxes()
        avg_time_prep_1hp += time.perf_counter() - time_prep_1hp

        # apply learned NN to predict the heat plumes
        time_apply_1nn = time.perf_counter()
        hp: HeatPumpBox
        for hp in single_hps:
            hp.primary_temp_field = hp.apply_nn(model_1HP)
            # save predicted Temp field as input for training as well
            hp.primary_temp_field = domain.reverse_norm(
                hp.primary_temp_field, property="Temperature [C]"
            )
        avg_time_apply_1nn += time.perf_counter() - time_apply_1nn

        time_prep_2hp = time.perf_counter()
        for hp in single_hps:
            hp.get_other_temp_field(single_hps)

        for hp in single_hps:
            hp.primary_temp_field = domain.norm(
                hp.primary_temp_field, property="Temperature [C]"
            )
            hp.other_temp_field = domain.norm(
                hp.other_temp_field, property="Temperature [C]"
            )
        avg_time_prep_2hp += time.perf_counter() - time_prep_2hp


        # apply 2HP-NN
        time_apply_2nn = time.perf_counter()
        for hp in single_hps:
            if not hp.other_temp_field.all() == 10.6:
                hp.output = hp.apply_nn(model_2HP, inputs="interim_outputs")
                hp.output = domain.reverse_norm(hp.output, property="Temperature [C]")

                if False:
                    hp.label = domain.reverse_norm(hp.label, property="Temperature [C]")
                    hp.save_pred(run_id, pathlib.Path(os.getcwd(), "runs", domain_prepared_dir.name), inputs_all=[hp.output])

                domain.add_hp(hp, hp.output)
            else:
                # if no overlap: dont apply 2HP-NN and just use 1HP-NN
                domain.add_hp(hp, hp.primary_temp_field)
                print("I didnt expect this to happen")
        avg_time_apply_2nn += time.perf_counter() - time_apply_2nn

        time_measure_and_visualize = time.perf_counter()
        for id_hp, hp in enumerate(single_hps):
            loss_mae, loss_mse = hp.measure_accuracy(domain, plot_args=[False, destination_2hp_nn / f"plot_hp{num_hps_overall}.png"])
            avg_mae[id_hp] += loss_mae
            avg_mse[id_hp] += loss_mse
            num_split[id_hp] += 1
            avg_loss_mae += loss_mae
            avg_loss_mse += loss_mse
            num_hps_overall += 1
        avg_time_measure_and_visualize += time.perf_counter() - time_measure_and_visualize
        domain.plot("t", folder=destination_2hp_nn, name=run_id)
        num_dp_valid += 1
        # domain.save(destination_2hp_nn, run_id)
    
    # avg measurements
    avg_time_prep_1hp /= num_dp_valid
    avg_time_apply_1nn /= num_dp_valid
    avg_time_prep_2hp /= num_hps_overall
    avg_time_apply_2nn /= num_hps_overall
    avg_time_measure_and_visualize /= num_hps_overall
    avg_loss_mae /= num_hps_overall
    avg_loss_mse /= num_hps_overall
    for id_hp in avg_mae.keys():
        avg_mae[id_hp] /= num_split[id_hp]
        avg_mse[id_hp] /= num_split[id_hp]
        # TODO after reasonable training: check, if still avg_x[0] so different to avg_x[1]
        # if not: remove the whole part about avg_mae and avg_mse and num_split

    with open(destination_2hp_nn / "measurements_apply.yaml", "w") as file:
        yaml.safe_dump(
            {
                "time whole process in sec": time.perf_counter() - time_begin,
                "timestamp_begin": timestamp_begin,
                "timestamp_end": time.ctime(),
                "avg_time_prep_1hp in sec (per dp)": avg_time_prep_1hp,
                "avg_time_apply_1nn (incl. renorming) in sec (per dp)": avg_time_apply_1nn,
                "avg_time_prep_2hp (incl. norming) in sec (per hp)": avg_time_prep_2hp,
                "avg_time_apply_2nn (incl. renorming) in sec (per hp)": avg_time_apply_2nn,
                "avg_time_measure_and_visualize in sec (per hp)": avg_time_measure_and_visualize,
                "time to prepare paths etc. in sec": time_prepare,
                "number valid datapoints": num_dp_valid,
                "avg_loss_mae": float(avg_loss_mae),
                "avg_loss_mse": float(avg_loss_mse),
                "number of heat pump boxes in training": num_hps_overall,
                "avg_mae": {int(k): float(v) for k, v in avg_mae.items()},
                "avg_mse": {int(k): float(v) for k, v in avg_mse.items()},
                "num_split": {int(k): int(v) for k, v in num_split.items()},
            },
            file,
        )
    
    with open(destination_2hp_nn / "args.yaml", "w") as file:
        yaml.safe_dump(
            {
                "dataset_large_name": dataset_large_name,
                "preparation_case": preparation_case,
                "model_name_2HP": model_name_2HP,
                "device": device,
            },
            file,
        )

    logging.warning(f"Finished applying 2HP-NN to {num_dp_valid} datapoints, results are at {destination_2hp_nn}")


def pipeline_apply_1HPNN(
    dataset_large_name: str,
    preparation_case: str,
    model_name_1HP: str = None,
    device: str = "cuda:0",
):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

    # prepare large dataset if not done yet
    (datasets_raw_domain_dir, datasets_prepared_domain_dir, domain_prepared_dir, datasets_model_trained_with_path, model_1hp_path, _, _, inputs_prep
     ) = set_paths_2hpnn(dataset_large_name, preparation_case)
    
    if not os.path.exists(domain_prepared_dir):
        args = SettingsPrepare(
            raw_dir=datasets_raw_domain_dir,
            datasets_dir=datasets_prepared_domain_dir,
            dataset_name=dataset_large_name,
            inputs_prep=inputs_prep,
            # name_extension=name_extension,
        )
        prepare_dataset(args,
            power2trafo=False,
            info=load_yaml(datasets_model_trained_with_path, "info"),
        )  # norm with data from dataset that NN was trained with!
        print(f"Domain {domain_prepared_dir} prepared")
    else:
        print(f"Domain {domain_prepared_dir} already prepared")

    destination_2hp_nn = pathlib.Path(os.getcwd(), "runs", domain_prepared_dir.name)
    destination_2hp_nn.mkdir(exist_ok=True)

    # load models
    model_1HP = UNet(in_channels = len(inputs_prep))
    model_1HP.load(model_1hp_path, device)
    
    num_dp_valid = 0
    avg_time_prep_1hp = 0
    avg_time_apply_1nn = 0
    avg_time_prep_2hp = 0
    avg_time_apply_2nn = 0
    avg_time_measure_and_visualize = 0
    num_hps_overall = 0
    avg_loss_mae = 0
    avg_loss_mse = 0
    avg_mae = {0: 0, 1: 0}
    avg_mse = {0: 0, 1: 0}
    num_split = {0: 0, 1: 0}

    time_prepare = time.perf_counter() - time_begin

    # apply 1HP-NN and 2HP-NN
    list_runs = os.listdir(os.path.join(domain_prepared_dir, "Inputs"))
    for run_file in tqdm(list_runs, desc="Apply 2HP-NN", total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        time_prep_1hp = time.perf_counter()
        domain = Domain(domain_prepared_dir, stitching_method="max", file_name=run_file)
        # generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        # generate 1hp-boxes and extract information like perm and ids etc.
        single_hps = domain.extract_hp_boxes()
        avg_time_prep_1hp += time.perf_counter() - time_prep_1hp

        # apply learned NN to predict the heat plumes
        time_apply_1nn = time.perf_counter()
        hp: HeatPumpBox
        for hp in single_hps:
            hp.primary_temp_field = hp.apply_nn(model_1HP)
            # save predicted Temp field as input for training as well
            hp.primary_temp_field = domain.reverse_norm(
                hp.primary_temp_field, property="Temperature [C]"
            )
        # avg_time_apply_1nn += time.perf_counter() - time_apply_1nn

        # time_prep_2hp = time.perf_counter()
        # for hp in single_hps:
            # hp.get_inputs_for_2hp_nn(single_hps)

        # for hp in single_hps:
            # hp.prediction_1HP = domain.norm(hp.prediction_1HP, property="Temperature [C]")
            # hp.interim_outputs = domain.norm(hp.interim_outputs, property="Temperature [C]")
        # avg_time_prep_2hp += time.perf_counter() - time_prep_2hp

            domain.add_hp(hp, hp.primary_temp_field)
        # avg_time_apply_2nn += time.perf_counter() - time_apply_2nn

        # time_measure_and_visualize = time.perf_counter()
        # for id_hp, hp in enumerate(single_hps):
        #     loss_mae, loss_mse = hp.measure_accuracy(domain, plot_args=[False, destination_2hp_nn / f"plot_hp{num_hps_overall}.png"])
        #     avg_mae[id_hp] += loss_mae
        #     avg_mse[id_hp] += loss_mse
        #     num_split[id_hp] += 1
        #     avg_loss_mae += loss_mae
        #     avg_loss_mse += loss_mse
        #     num_hps_overall += 1
        # avg_time_measure_and_visualize += time.perf_counter() - time_measure_and_visualize
        domain.plot("goksit", folder=destination_2hp_nn, name=run_id)
        num_dp_valid += 1
        domain.save(destination_2hp_nn, run_id)
    
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

    # with open(destination_2hp_nn / "measurements_apply.yaml", "w") as file:
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
    
    # with open(destination_2hp_nn / "args.yaml", "w") as file:
    #     yaml.safe_dump(
    #         {
    #             "dataset_large_name": dataset_large_name,
    #             "preparation_case": preparation_case,
    #             "model_name_2HP": model_name_2HP,
    #             "device": device,
    #         },
    #         file,
    #     )

    # logging.warning(f"Finished applying 2HP-NN to {num_dp_valid} datapoints, results are at {destination_2hp_nn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation_case", type=str, default="gksi_100dp")
    parser.add_argument("--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm")
    parser.add_argument("--model_2hp", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    assert args.preparation_case in ["gksi_100dp", "gksi_1000dp", "pksi_100dp", "pksi_1000dp"], "preparation_case must be one of ['gksi_100dp', 'gksi_1000dp', 'pksi_100dp', 'pksi_1000dp']"

    pipeline_apply_2HPNN(
        dataset_large_name=args.dataset_large,
        preparation_case=args.preparation_case,
        model_name_2HP=args.model_2hp,
        device=args.device,
    )

    # pipeline_apply_1HPNN(
    #     dataset_large_name=args.dataset_large,
    #     preparation_case=args.preparation_case,
    #     device=args.device,
    # )