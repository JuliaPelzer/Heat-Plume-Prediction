import gc
import logging
import multiprocessing
from code.postprocessing.visualization import visualize_inputs, visualize_outputs
from code.preprocessing.data_init import construct_dataloader, init_data
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.preprocessing import preprocessing
from code.processing.loss_fcts import CombiLoss
from code.processing.networks.convLSTM import Seq2Seq
from code.processing.networks.unetVariants import UNet, UNetNoPad2
from code.processing.solver import Solver
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import (
    check_model_avail,
    get_data_prep_path,
    load_time_steps,
    save_yaml,
)
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import HuberLoss, L1Loss, MSELoss


def get_dataset_type(info: dict) -> str:
    datasetType = None
    temp_label = info.get("Labels", {}).get("Temperature [C]")

    if temp_label is None:
        datasetType = DatasetType.unknown
    elif 10.6 <= temp_label["min"]:
        datasetType = DatasetType.steady_state_heating
    elif temp_label["max"] <= 10.6:
        datasetType = DatasetType.steady_state_cooling
    else:
        datasetType = DatasetType.seasonal
    return datasetType


def training(args: dict):
    multiprocessing.set_start_method("spawn", force=True)

    args["data_prep"] = get_data_prep_path(args["data_prep"], args["inputs"], args["outputs"], args["data_raw"])
    info = preprocessing(args)  # and save info.yaml in model folder
    datasetType = get_dataset_type(info)

    input_channels, output_channels, datasets = init_data(
        args,
        datapoint_test=args["datapoint_test"],
        datapoint_validate=args["datapoint_validate"],
        datapoint_train=args["datapoint_train"],
        tmp_bool_cutouts=args["bool_cutouts"],
    )

    dataloaders = {}
    dataloaders["train"] = construct_dataloader(args["batchsize"], datasets["train"], shuffle=True)
    dataloaders["val"] = construct_dataloader(args["batchsize"], datasets["val"], shuffle=False)
    dataloaders["test"] = construct_dataloader(args["batchsize"], datasets["test"], shuffle=False)

    # visualize only inputs
    if args["visualize"]:
        visualize_inputs(
            datasetType,
            dataloaders["val"],
            args,
            plot_path=args["destination"] / "val",
            amount_datapoints_to_visu=1,
            pic_format="png",
        )
        visualize_inputs(
            datasetType,
            dataloaders["test"],
            args,
            plot_path=args["destination"] / "test",
            amount_datapoints_to_visu=1,
            pic_format="png",
        )

    # Step 1
    if "t" not in args["outputs"]:
        log.info("Using UNet for step 1")
        model = UNet(
            in_channels=input_channels,
            out_channels=output_channels,
            depth=args["depth"],
            init_features=args["init_features"],
            kernel_size=args["kernel_size"],
            stride=args["stride"],
            dilation=args["dilation"],
            activation=args["activation_fct"],
            norm=args["norm"],
            repeat_inner=args["repeat_inner"],
        ).float()
    # Step 3
    else:
        network = args.get("network", "unet").lower()
        if network in ["convlstm", "rnn", "lstm"]:
            log.info("Using ConvLSTM for step 3")
            nr_time_steps = load_time_steps(
                Path(args["data_raw"], "RUN_" + str(args["datapoint_train"]), "pflotran.h5")
            )
            len_box = args["len_box"]
            model = Seq2Seq(
                input_channels,
                frame_size=[len_box, len_box],
                prev_boxes=0,
                extend=len(nr_time_steps) - 2,
                num_layers=1,
                enc_conv_features=[2**i for i in range(4, 8)],
                dec_conv_features=list(reversed([2**i for i in range(4, 8)])),
                enc_kernel_sizes=[5 for i in range(4)],
                dec_kernel_sizes=[5 for i in range(4)],
            ).float()
        else:
            log.info("Using UNet for step 3")
            model = UNetNoPad2(
                in_channels=input_channels,
                out_channels=output_channels,
                depth=args["depth"],
                init_features=args["init_features"],
                kernel_size=args["kernel_size"],
                stride=args["stride"],
                dilation=args["dilation"],
                activation=args["activation_fct"],
                norm=args["norm"],
                repeat_inner=args["repeat_inner"],
            ).float()

    model.to(args["device"])

    if args["case"] in ["test", "finetune"]:
        check_model_avail(args)
        model.load(args["model"], args["device"])

    is_pretrained = args["case"] in ["finetune", "test"]
    val_loss = 9e9
    if args["case"] in ["train", "finetune"]:
        loss = select_loss_function(args)
        solver = Solver(
            model,
            datasets["train"],
            datasets["val"],
            loss_func=loss,
            finetune=is_pretrained,
            optimizer_switch=args["optimizer_switch"],
            batchsize=args["batchsize"],
        )
        training_time = datetime.now()
        try:
            val_loss = solver.train(datasetType, dataloaders["train"], dataloaders["val"], args)
        except KeyboardInterrupt:
            if solver.best_model_params is not None:
                logging.warning(
                    f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}."
                )
                val_loss = solver.best_model_params["loss"]
            else:
                logging.warning("Manually stopping training early. No best model found yet.")
        finally:
            log.info("Training finished")

        # save model
        training_time = datetime.now() - training_time
        model.save(args["destination"])
        solver.save_metrics_separate_yaml(
            dataloaders,
            args["destination"],
            args["device"],
            {
                "best_epoch": solver.best_model_params["epoch"] if solver.best_model_params is not None else -1,
                "no_params": model.num_of_params(),
                "max_epochs": args["epochs"],
                "training_time": training_time.total_seconds(),
            },
        )
    elif args["case"] == "test":
        model.eval()
        loss = select_loss_function(args)
        solver = Solver(
            model,
            datasets["train"],
            datasets["val"],
            loss_func=loss,
            finetune=is_pretrained,
            optimizer_switch=args["optimizer_switch"],
            batchsize=args["batchsize"],
        )
        solver.save_metrics_separate_yaml(dataloaders, args["destination"], args["device"], {})

    # postprocessing, visualize only outputs
    if args["visualize"]:
        visualize_outputs(
            datasetType,
            model,
            dataloaders["val"],
            args,
            plot_path=args["destination"] / "val",
            amount_datapoints_to_visu=1,
            pic_format="png",
        )
        visualize_outputs(
            datasetType,
            model,
            dataloaders["test"],
            args,
            plot_path=args["destination"] / "test",
            amount_datapoints_to_visu=1,
            pic_format="png",
        )

    for key in ["train", "val", "test"]:
        if hasattr(dataloaders[key], "_iterator") and dataloaders[key]._iterator is not None:
            try:
                dataloaders[key]._iterator._shutdown_workers()
            except Exception:
                pass  # Prevent error masking if they are already dead
        del dataloaders[key]
    gc.collect()

    # Clear up memory
    del model
    del datasets
    torch.cuda.empty_cache()

    return val_loss


def select_loss_function(args):
    if args["train_loss"].lower() == "mae":
        loss = L1Loss()
    elif args["train_loss"].lower() == "mse":
        loss = MSELoss()
    elif args["train_loss"].lower() == "huber":
        loss = HuberLoss()
    elif args["train_loss"].lower() == "combi":
        loss = CombiLoss(0.75)
    return loss
