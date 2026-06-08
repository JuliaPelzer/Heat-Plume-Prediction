import gc
import logging
import multiprocessing
from code.postprocessing.visualization import visualize_inputs, visualize_outputs
from code.preprocessing.data_init import construct_dataloader, init_data
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.preprocessing import preprocessing
from code.preprocessing.validity_checks import receptive_field_is_sufficient
from code.processing.loss_fcts import (
    BinaryCrossEntropy,
    CombiLoss,
    CombinedFocalMSE,
    FocalMAE,
    FocalMAE_FP,
    FocalMSE,
    FocalMSE_FP,
    MAE_Logits,
    SSIMLoss,
    WeightedMSE,
)
from code.processing.networks.convLSTM import Seq2Seq
from code.processing.networks.unetVariants import UNet, UNetNoPad2
from code.processing.solver import Solver
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import (
    check_model_avail,
    get_data_prep_path,
    load_time_steps,
)
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import HuberLoss, L1Loss, MSELoss
from torch.utils.data import DataLoader


def save_predictions(model, dataset, result_destination: Path, args: dict):
    """Run inference on dataset and save predictions as individual .pt files"""
    device = args["device"]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Run inference
        if dataset.__class__.__name__ in ["SimulationDatasetCutsSequential", "DataPointSequence", "Subset"]:
            if inputs.shape[-1] > 500 or inputs.shape[-2] > 500:
                y_out = model.infer_tiled(inputs, device)
            else:
                y_out = model.infer(inputs, device)
        else:
            y_out = model.infer(inputs, device)

        # Save individual prediction and preserve run id naming
        prediction = y_out.detach().cpu().squeeze(0)
        if hasattr(dataset, "input_names") and batch_idx < len(dataset.input_names):
            filename = Path(dataset.input_names[batch_idx]).name
        else:
            filename = f"RUN_{batch_idx}.pt"
        torch.save(prediction, result_destination / filename)

        if batch_idx % 10 == 0:
            log.info(f"Saved prediction {batch_idx + 1}/{len(dataloader)} as {filename}")

    log.info(f"Saved {len(dataloader)} predictions to {result_destination}")


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
        bool_cutouts=args["bool_cutouts"],
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
        if args["network"] in ["convlstm", "rnn", "lstm"]:
            log.info("Using ConvLSTM for current step")
            if "time_steps_to_predict" in args and args["time_steps_to_predict"] is not None:
                time_steps_to_predict = args["time_steps_to_predict"]
            else:
                time_steps_to_predict = load_time_steps(
                    Path(args["data_raw"], "RUN_" + str(args["order_data"][0]), "pflotran.h5")
                )
            # Support nested lists for sublist scheduling
            if (
                isinstance(time_steps_to_predict, list)
                and len(time_steps_to_predict) > 0
                and isinstance(time_steps_to_predict[0], list)
            ):
                if len(set(len(sub) for sub in time_steps_to_predict)) != 1:
                    raise ValueError("All time_steps_to_predict sublists must have the same length")
                extend = len(time_steps_to_predict[0])
            else:
                extend = len(time_steps_to_predict)

            log.info(f"Extend: {extend}")
            len_box = args["len_box"]
            log.info(f"Input_channels before net construction: {input_channels}")
            model = Seq2Seq(
                input_channels + 2,
                frame_size=[len_box, len_box],
                prev_boxes=0,
                extend=extend,
                num_layers=args["num_layers"],
                enc_conv_features=args["enc_conv_features"],
                dec_conv_features=args["dec_conv_features"],
                enc_kernel_sizes=args["enc_kernel_sizes"],
                dec_kernel_sizes=args["dec_kernel_sizes"],
            ).float()
            receptive_field_dict = model.calculate_receptive_field()
            if not receptive_field_is_sufficient(receptive_field_dict["total_rf"], dataloaders):
                log.warning(
                    f"Warning: ConvLSTM receptive field {receptive_field_dict['total_rf']} is smaller than the required input size."
                )
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

    is_pretrained = args["case"] in ["finetune", "test"]
    val_loss = 9e9
    loss = select_loss_function(args)
    log.info(f"Loss function selected: {loss.__class__.__name__}")
    solver = Solver(
        model,
        datasets["train"],
        datasets["val"],
        loss_func=loss,
        finetune=is_pretrained,
        optimizer_switch=args["optimizer_switch"],
        batchsize=args["batchsize"],
    )
    if args["case"] in ["test", "finetune"]:
        check_model_avail(args)
        model.load(args["model"], args["device"])

    if args["case"] in ["train", "finetune"]:
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
        log.info(f"Number of test datapoints: {len(dataloaders['test'].dataset)}")
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
    match args["train_loss"].lower():
        case "mae":
            return L1Loss()
        case "mse":
            return MSELoss()
        case "weightedmse":
            return WeightedMSE()
        case "huber":
            return HuberLoss()
        case "combi":
            return CombiLoss(0.75)
        case "ssim":
            return SSIMLoss()
        case "focalmse":
            return FocalMSE()
        case "combined_focalmse":
            return CombinedFocalMSE()
        case "bce":
            return BinaryCrossEntropy()
        case "focalmse_fp":
            return FocalMSE_FP()
        case "focalmae_fp":
            return FocalMAE_FP()
        case "focalmae":
            return FocalMAE()
        case "mae_logits":
            return MAE_Logits()
        case _:
            raise ValueError(f"Unknown train_loss: {args['train_loss']}")
