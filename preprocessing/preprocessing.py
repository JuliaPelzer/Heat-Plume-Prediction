from copy import deepcopy
from pathlib import Path
import torch
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from preprocessing.prepare_dataset import prepare_dataset, is_unprepared
from preprocessing.domain_classes.domain import Domain, DomainBatch
from preprocessing.domain_classes.heat_pump import HeatPumpBox, apply_nn_batch
from processing.networks.model import Model
from processing.networks.unet import UNet
from processing.networks.CdMLP import CdMLP
from processing.networks.unetVariants import UNetHalfPad2
# import processing.pipelines.extend_plumes as ep
import utils.utils_args as ua


def preprocessing(args:dict):
    print("Preparing dataset")
    # handling of case=="test"? TODO
    if is_unprepared(args["data_prep"]):
        if args["problem"] == "2stages":
            exit("2stages not implemented yet, use other branch")

        if args["problem"] == "allin1" and "n" in args["inputs"]: # case: different datasets
            # TODO handling of different datasets and several stages
            additional_inputs_unnormed = preprocessing_allin1_v2(args)
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
    case_1hpnn = args["allin1_prepro_n_case"]

    run_ids = ua.get_run_ids_from_raw(args["data_raw"])
    additional_inputs = []
    assert len(run_ids) == 3, "allin1 requires 3 runs for train, val, test"

    args_1hpnn = set_args_1hpnn(case_1hpnn)
    args_extend = set_extend_args()    

    args_domain_with_1hpnn_params["inputs"] = args_1hpnn["inputs"]
    args_domain_with_1hpnn_params["model"] = args_1hpnn["model"]
    args_domain_with_1hpnn_params["data_prep"] = args["data_raw"].name + " inputs_" + args_1hpnn["inputs"]
    args_domain_with_1hpnn_params["destination"] = None #irrelevant
    ua.make_paths(args_domain_with_1hpnn_params, make_model_and_destination_bool=False)

    for run_id in tqdm(run_ids, desc="Runs"):
        preprocessing_dir = args_domain_with_1hpnn_params["data_prep"] / f"Preprocessed {case_1hpnn}"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_destination = preprocessing_dir / f"RUN_{run_id}_n.pt"

        if (preprocessing_destination).exists():
            print(f"Loading domain prediction from 1hpnn+ep from {preprocessing_destination}")
            print(args["device"])
            additional_input = torch.load(preprocessing_destination, map_location=args["device"])
            additional_input = additional_input.detach()

        else:
            print(f"Preparing domain for allin1 with {case_1hpnn}+(ep)")

            model_1hp, info_1hp = load_1hp_model_and_info(case_1hpnn, args_1hpnn, args["device"])

            # prepare allin1 domain with 1hp-model normalization for cutting out hp boxes
            if is_unprepared(args_domain_with_1hpnn_params["data_prep"]): # or args.case == "test":
                print(f"Preparing domain for preprocessing at {args_domain_with_1hpnn_params['data_prep']}")
                prepare_dataset(args_domain_with_1hpnn_params, info=info_1hp) # TODO make faster by ignoring "s" in prep and adding dummy dimension afterwards


            # extract hp boxes
            domain = Domain(args_domain_with_1hpnn_params["data_prep"], stitching_method="max", file_name=f"RUN_{run_id}.pt", device=args["device"])
            threshold_T = domain.norm(10.7, property = "Temperature [C]")
            single_hps, hps_inputs = domain.extract_hp_boxes(args["device"])

            hp: HeatPumpBox
            # for hp in tqdm(single_hps, desc="Applying 1HPNN + ep"):
            #     if case_1hpnn == "gt":
            #         hp.primary_temp_field = hp.label.squeeze(0)
            #     else:
            #         hp.primary_temp_field = hp.apply_nn(case_1hpnn, model_1hp, device=args["device"], info=info_1hp) # TODO prediction is shitty -> check for better 1hp-nn

            #     # extend plumes
            #     # while hp.primary_temp_field[-1].max() < threshold_T:
            #     #     hp.extend_plume()
                
            #     domain.add_hp(hp)

            # matrix of all extracted hp boxes for nn
            if case_1hpnn == "gt":
                predictions = torch.stack([hp.label.squeeze() for hp in single_hps])
            else:
                predictions = apply_nn_batch(case_1hpnn, hps_inputs, model_1hp, device=args["device"])

            for hp, prediction in tqdm(zip(single_hps, predictions), desc="Adding box-predictions (+ep) to domain"):
                hp.primary_temp_field = prediction
                domain.add_hp(hp)
            

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
            plt.show()

            if len(domain.prediction.shape) == 2:
                domain.prediction = domain.prediction.unsqueeze(0)
            additional_input = domain.prediction.detach().cpu()
            print(f"Saving domain to {preprocessing_destination}")
            torch.save(additional_input, preprocessing_destination)
            ua.save_yaml({"case 1hpnn": case_1hpnn, "1hp": args_1hpnn, "extend": args_extend}, preprocessing_destination.parent / "args.yaml")

        additional_inputs.append(additional_input) # TODO better as dict?
    return additional_inputs

def preprocessing_allin1_v2(args: dict):
    args_domain_with_1hpnn_params = deepcopy(args)
    case_1hpnn = args["allin1_prepro_n_case"]

    run_ids = ua.get_run_ids_from_raw(args["data_raw"])
    additional_inputs = []
    assert len(run_ids) == 3, "allin1 requires 3 runs for train, val, test"

    args_1hpnn = set_args_1hpnn(case_1hpnn)
    args_extend = set_extend_args()    

    args_domain_with_1hpnn_params["inputs"] = args_1hpnn["inputs"]
    args_domain_with_1hpnn_params["model"] = args_1hpnn["model"]
    args_domain_with_1hpnn_params["data_prep"] = args["data_raw"].name + " inputs_" + args_1hpnn["inputs"]
    args_domain_with_1hpnn_params["destination"] = None #irrelevant
    ua.make_paths(args_domain_with_1hpnn_params, make_model_and_destination_bool=False)

    for run_id in tqdm(run_ids, desc="Runs"):
        preprocessing_dir = args_domain_with_1hpnn_params["data_prep"] / f"Preprocessed {case_1hpnn}"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_destination = preprocessing_dir / f"RUN_{run_id}_n.pt"

        if (preprocessing_destination).exists():
            print(f"Loading domain prediction from 1hpnn+ep from {preprocessing_destination}")
            additional_input = torch.load(preprocessing_destination, map_location="cuda:0")
            additional_input = additional_input.detach().to("cpu")

        else:
            print(f"Preparing domain for allin1 with {case_1hpnn}+(ep)")

            model_1hp, info_1hp = load_1hp_model_and_info(case_1hpnn, args_1hpnn, args["device"])

            # prepare allin1 domain with 1hp-model normalization for cutting out hp boxes
            if is_unprepared(args_domain_with_1hpnn_params["data_prep"]): # or args.case == "test":
                print(f"Preparing domain for preprocessing at {args_domain_with_1hpnn_params['data_prep']}")
                prepare_dataset(args_domain_with_1hpnn_params, info=info_1hp) # TODO make faster by ignoring "s" in prep and adding dummy dimension afterwards


            # extract hp boxes
            domain = DomainBatch(args_domain_with_1hpnn_params["data_prep"], stitching_method="max", file_name=f"RUN_{run_id}.pt", device=args["device"])
            threshold_T = domain.norm(10.7, property = "Temperature [C]")
            hps, relative_pos_hp = domain.extract_hp_boxes(args["device"])
            # matrix of all extracted hp boxes for nn
            if case_1hpnn == "gt":
                hps["predictions"] = hps["labels"]
            else:
                 hps["predictions"] = apply_nn_batch(case_1hpnn, hps["inputs"], model_1hp, device=args["device"])

            idi = 15
            plt.subplot(2,3,1)
            plt.imshow(hps["predictions"][idi].detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2,3,2)
            plt.imshow(hps["labels"][idi].detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2,3,3)
            plt.imshow(hps["inputs"][idi][0].detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2,3,4)
            plt.imshow(hps["inputs"][idi][1].detach().cpu().numpy().T)
            plt.colorbar()
            plt.subplot(2,3,5)
            plt.imshow(hps["inputs"][idi][2].detach().cpu().numpy().T)
            plt.colorbar()
            # plt.subplot(2,3,6)
            # plt.imshow(hps["inputs"][idi][3].detach().cpu().numpy().T)
            # plt.colorbar()
            plt.show()
            maxs = []
            mins = []
            for ii in range(len(hps["inputs"])):
                maxii = hps["inputs"][ii][1].max().detach().cpu().numpy()
                minii = hps["inputs"][ii][1].min().detach().cpu().numpy()
                maxs.append(maxii)
                mins.append(minii)
                print(ii, maxii, minii)
            print(max(maxs), min(mins))
            exit()

            # TODO rel-pos wrong way around? # TODO why so slow?
            domain.add_predictions(hps, relative_pos_hp)
            
            if len(domain.prediction.shape) == 2:
                domain.prediction = domain.prediction.unsqueeze(0)
            additional_input = domain.prediction.detach().to("cpu")
            print(f"Saving domain to {preprocessing_destination}")
            torch.save(additional_input, preprocessing_destination)
            ua.save_yaml({"case 1hpnn": case_1hpnn, "1hp": args_1hpnn, "extend": args_extend}, preprocessing_destination.parent / "args.yaml")

        additional_inputs.append(additional_input) # TODO better as dict?
    return additional_inputs

def load_1hp_model_and_info(case_1hpnn:str, args_1hpnn: dict, device: str):
    if case_1hpnn == "unet":
        model_1hp:Model = args_1hpnn["model_type"](len(args_1hpnn["inputs"]))
        model_1hp.load(args_1hpnn["model"], device)
        model_1hp.to(device)

    elif case_1hpnn == "cdmlp":
        assert args_1hpnn["inputs"] == "gksi", "CdMLP only implemented for gksi inputs."
        model_1hp:CdMLP = args_1hpnn["model_type"](args_1hpnn["model"]/"model.keras")

    else:
        model_1hp = None

    info = ua.load_yaml(args_1hpnn["model"]/"info.yaml")
    return model_1hp, info


def load_extend_model(args_extend: dict, device: str):
    model_ep:Model = args_extend["model_type"](len(args_extend["inputs"])+1)
    model_ep.load(args_extend["model"], device)
    model_ep.to(device)
    return model_ep


def correct_skip_in_field(args_extend, actual_len):
    if actual_len < args_extend["skip_in_field"]: 
        skip_in_field = actual_len
        print(f"Changed skip_in_field to {skip_in_field} because actual_len is smaller ({actual_len}).")
    
    return args_extend

def set_args_1hpnn(case:str):
    if case in ["unet", "gt"]:
        args_1hpnn = {
            # "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_10000dp_varyK_v3_part2 inputs_gksi box256 skip256"), 
            "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_10000dp_varyK_v3_part1 inputs_sik box256 skip256"),
            #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_10000dp_varyK_V2 inputs_gksi box256 skip32"), 
            #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/dataset_small_1000dp_varyK_v2 inputs_gksi case_train box256 skip32"), 
            #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hp/vary_k/dataset_medium_100dp_vary_perm inputs_gksi case_train box256 skip256 UNet"),
            "model_type": UNet,
            "inputs": "sik", #"gksi", #
            }
    elif case == "cdmlp":
        args_1hpnn = {
            "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/cdmlp/runs/pump_indices augmented full mean"),
            "model_type": CdMLP,
            "inputs": "gksi",
        }
    return args_1hpnn

def set_extend_args():
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

    return args_extend