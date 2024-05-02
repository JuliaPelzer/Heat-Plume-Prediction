from copy import deepcopy
from pathlib import Path
import torch
from tqdm.auto import tqdm

from preprocessing.prepare_dataset import prepare_dataset, is_unprepared
from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from processing.networks.model import Model
from processing.networks.unet import UNet
from processing.networks.CdMLP import CdMLP
from processing.networks.unetVariants import UNetHalfPad2
# import processing.pipelines.extend_plumes as ep
import utils.utils_args as ua


def preprocessing(args:dict):
    print("Preparing dataset")
    # handling of case=="test"? TODO
    if is_unprepared(args["data_prep"]): # or args.case == "test":
        if args["problem"] == "2stages":
            exit("2stages not implemented yet, use other branch")

        if args["problem"] == "allin1" and "n" in args["inputs"]: # case: different datasets
            # TODO handling of different datasets and several stages
            additional_inputs_unnormed = preprocessing_allin1(args)
            # TODO several runs now!
        else: 
            additional_inputs_unnormed = None

        info = ua.load_yaml(args["model"]/"info.yaml") if args["case"] != "train" else None
        info = prepare_dataset(args, info=info, additional_inputs=additional_inputs_unnormed)
            # handling of case=="test"? TODO

    else:
        info = ua.load_yaml(args["data_prep"]/"info.yaml") 
    print(f"Dataset prepared: {args['data_prep']}")

    if args["case"] == "train": # TODO also finetune?
        ua.save_yaml(info, args["model"]/"info.yaml")

def preprocessing_allin1(args: dict):
    args_domain_with_1hpnn_params = deepcopy(args)

    run_ids = ua.get_run_ids_from_raw(args["data_raw"])
    additional_inputs = []
    assert len(run_ids) == 3, "allin1 requires 3 runs for train, val, test"

    for run_id in tqdm(run_ids, desc="Runs"):
        preprocessing_dir = args["data_prep"] / "Preprocessed"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_destination = preprocessing_dir / f"RUN_{run_id}_n.pt"


        if (preprocessing_destination).exists():
            print(f"Loading domain prediction from 1hpnn+ep from {preprocessing_destination}")
            # load prediction (1hpnn+ep)-file if exists
            additional_input = torch.load(preprocessing_destination)
            additional_input = additional_input.detach()
            # TODO CHECK this scenario

        else:
            print("Preparing domain for allin1")
            # preprocessing with neural network: 1hpnn(+extend_plumes) or with groundtruth or with cdmlp
            case_1hpnn = "unet"

            if case_1hpnn == "unet":
                args_1hpnn = {
                    "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_10000dp_varyK_v3_part1 inputs_sik box256 skip256"),
                    #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_10000dp_varyK_V2 inputs_gksi box256 skip32"), 
                    #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_1000dp_varyK_v2 inputs_gksi case_train box256 skip32"), 
                    #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/vary_k/dataset_medium_100dp_vary_perm inputs_gksi case_train box256 skip256 UNet"),
                    "model_type": UNet,
                    "inputs": "sik", #"gksi",
                    }
            elif case_1hpnn == "cdmlp":
                args_1hpnn = {
                    "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/cdmlp/runs/pump_indices augmented full mean"),
                    "model_type": CdMLP,
                    "inputs": "gksi",
                }

            model_1hp, info_1hp = load_1hp_model_and_info(case_1hpnn, args_1hpnn, args["device"])

            # args_extend = {
            #     "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend/vary_k/dataset_medium_100dp_vary_perm inputs_gk case_train box128 skip2"), #test_overlap_input_T"),
            #     "model_type": UNetHalfPad2,
            #     "inputs": "gk",
            #     "box_size": 128,
            #     "start_prior_box": 64, # box_size // 2
            #     "skip_per_dir": 4,
            #     "skip_in_field": 32, #< actual_len
            #     "overlap": 46, # manually chosen for UNetHalfPad2
            #     }
            # model_ep = load_extend_model(args_extend, args["device"])
            args_extend = {}

            # prepare allin1 domain with 1hp-model normalization for cutting out hp boxes
            args_domain_with_1hpnn_params["inputs"] = args_1hpnn["inputs"]
            args_domain_with_1hpnn_params["model"] = args_1hpnn["model"]
            args_domain_with_1hpnn_params["data_prep"] = args["data_raw"].name + " inputs_" + args["inputs"] + " " + case_1hpnn # TODO test
            args_domain_with_1hpnn_params["destination"] = None
            # args_gksi.destination should be irrelevant because no model is trained
            ua.make_paths(args_domain_with_1hpnn_params, make_model_and_destination_bool=False)
            if is_unprepared(args_domain_with_1hpnn_params["data_prep"]): # or args.case == "test":
                prepare_dataset(args_domain_with_1hpnn_params, info=info_1hp) # TODO make faster by ignoring "s" in prep and adding dummy dimension afterwards

            # extract hp boxes
            print(args_domain_with_1hpnn_params["data_prep"])
            domain = Domain(args_domain_with_1hpnn_params["data_prep"], stitching_method="max", file_name=f"RUN_{run_id}.pt", device=args["device"])
            threshold_T = domain.norm(10.7, property = "Temperature [C]")
            single_hps = domain.extract_hp_boxes(args["device"])
            
            hp: HeatPumpBox
            for hp in tqdm(single_hps, desc="Applying 1HPNN + ep"):
                if case_1hpnn == "gt":
                    hp.primary_temp_field = hp.label.squeeze(0)
                else:
                    hp.primary_temp_field = hp.apply_nn(case_1hpnn, model_1hp, device=args["device"], info=info_1hp) # TODO prediction is shitty -> check for better 1hp-nn
                    plt.imshow(hp.primary_temp_field.detach().cpu().numpy().T)
                    plt.colorbar()
                    plt.savefig("test.png")
                    exit()
                    # TODO push through as batch to increase speed TODO TODO 
                    

                # extend plumes
                # while hp.primary_temp_field[-1].max() < threshold_T:
                #     hp.extend_plume()
                
                domain.add_hp(hp)

            import matplotlib.pyplot as plt
            plt.subplot(2, 2, 1)
            plt.imshow(domain.inputs[1].detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.imshow(domain.inputs[0].detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2, 2, 3)
            plt.imshow(domain.prediction.detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2, 2, 4)
            plt.imshow(domain.label.detach().cpu().numpy().T)
            plt.colorbar()
            plt.savefig("test.png")

            if len(domain.prediction.shape) == 2:
                domain.prediction = domain.prediction.unsqueeze(0)
            additional_input = domain.prediction.detach().cpu()
            print(f"Saving domain to {preprocessing_destination}")
            torch.save(domain.prediction, preprocessing_destination)
            ua.save_yaml({"1hp": args_1hpnn, "extend": args_extend}, preprocessing_destination.parent / "info.yaml")

        additional_inputs.append(additional_input) # TODO better as dict?
    return additional_inputs

def load_1hp_model_and_info(case_1hpnn:str, args_1hpnn: dict, device: str):
    if case_1hpnn == "unet":
        model_1hp:Model = args_1hpnn["model_type"](len(args_1hpnn["inputs"]))
        model_1hp.load(args_1hpnn["model"], map_location=device)
        model_1hp.to(device)
        info = ua.load_yaml(args_1hpnn["model"]/"info.yaml")

    elif case_1hpnn == "cdmlp":
        assert args_1hpnn["inputs"] == "gksi", "CdMLP only implemented for gksi inputs."
        model_1hp:CdMLP = args_1hpnn["model_type"](args_1hpnn["model"]/"model.keras")
        info = ua.load_yaml(args_1hpnn["model"]/"info.yaml")

    return model_1hp, info


def load_extend_model(args_extend: dict, device: str):
    model_ep:Model = args_extend["model_type"](len(args_extend["inputs"])+1)
    model_ep.load(args_extend["model"]) # for cpu maybe: ,map_location=torch.device(device))
    model_ep.to(device)
    return model_ep


def correct_skip_in_field(args_extend, actual_len):
    if actual_len < args_extend["skip_in_field"]: 
        skip_in_field = actual_len
        print(f"Changed skip_in_field to {skip_in_field} because actual_len is smaller ({actual_len}).")
    
    return args_extend