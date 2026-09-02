import torch
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import shutil
import argparse

from preprocess_for_step3.generate_wellflow_field import generate_wellflow_field
from preprocess_for_step3.renorm_dataset import renorm_dataset
from preprocess_for_step3.step1_with_residuals import step1_with_residuals
from step2_streamlines.streamlines_main import build_streamlines
from preprocessing.preprocessing import import_dataset

## For training:
    # after generate_wellflow_field, but no renorming, infering of step1 necessary...
    # if training: 
    #     # calc residuals, only possible if replace_v = False
    #     repl_src = "_wf_replace_v" if replace_v else "_wf_not_replace_v"
    #     src = src.parent / (src.name + repl_src)
    #     def unnorm(channel: torch.Tensor, stats: dict) -> torch.Tensor:
    #         return channel * (stats["max"] - stats["min"]) + stats["min"]

    #     def norm(channel: torch.Tensor, stats: dict) -> torch.Tensor:
    #         return (channel  - stats["min"]) / (stats["max"] - stats["min"])

    #     def stats_by_i(section: dict) -> dict:
    #         return {entry["index"]: entry for entry in section.values()}

    #     tmp_in = torch.load(src / "Inputs/Sim_4.pt")
    #     tmp_lab = torch.load(src / "Labels/Sim_4.pt")
    #     info = yaml.safe_load(open(src / "info.yaml"))
    #     info_in = stats_by_i(info["Inputs"])
    #     info_lab = stats_by_i(info["Labels"])
    #     for i in range(len(tmp_in)):
    #         tmp_in[i] = unnorm(tmp_in[i], info_in[i])

    #     for i in range(len(tmp_lab)):
    #         tmp_lab[i] = unnorm(tmp_lab[i], info_lab[i])

def main_infer(src:Path, model_v:Path, model_T:Path, replace_v:bool=False, src_npz:Path=None):
    '''
    Prepare dataset for inference of LGCNN model part 3 (predicting temperatures from end-time velocity fields and streamlines)
    '''
    # define paths for intermediate steps here, to check if they already exist and skip them if so
    repl_src = "_wf_replace_v" if replace_v else "_wf_not_replace_v"
    src_ren = src.parent / (src.name + repl_src)
    dest_ren = src_ren.parent / (src_ren.name + "_renormed")
    src_inf1 = dest_ren
    dest_inf1 = src_inf1.parent / (src_inf1.name + "_prededV")
    dest_stream = dest_inf1.parent / (dest_inf1.name + "+s(t,Q,Ttrend)")
    src_ren2 = dest_inf1.parent / (dest_inf1.name + "+s(t,Q,Ttrend)")
    dest_ren2 = src_ren2.parent / (src_ren2.name + "_renormed")
    new_info = model_v / "info.yaml"

    ## Check if src exists (basis in torch format), if not: import from original npz format
    if not src_ren.exists():
        print("Not yet in .pt format, importing from .npz")
        test_data = "TEST" in src.stem
        import_dataset(src_npz, src, test_data=test_data)
        print(f"Imported dataset from {src_npz} to {src}")
    else:
        print(f"Dataset already in .pt format at {src}")

    ## Calc additional inputs of "wellflow" := radial flow fields around pumps from pump inflow with option to replace the original v(t0)
    if not dest_ren.exists():
        generate_wellflow_field(src, replace_v=replace_v)
        print(f"Done calc-wf to {src_ren}")

    ##  Renorm Dataset acc. to Trained-On-Dataset
    if not dest_inf1.exists():
        renorm_dataset(src_ren, new_info, dest_ren)
        print(f"Done renorming to {dest_ren} for predicting v")

    ## Infer Step 1
    predicted_residuals = True
    step3b_case = "3b" in src.stem
    if not dest_stream.exists():
        step1_with_residuals(predicted_residuals, step3b_case, 3, 4, model_v, src_inf1, src, dest_inf1)
        print(f"Done inferring step 1 to {dest_inf1}")

    ## Apply Step 2
    if not dest_ren2.exists():
        build_streamlines(dest_inf1, method="Radau")
        print(f"Done calc streamlines to {dest_stream}")

        ## Renorm Dataset acc. to Trained-On-Dataset for predicting T
        renorm_dataset(src_ren2, model_T / "info.yaml", dest_ren2)
        print(f"Done renorming to {dest_ren2} for predicting T")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_pt", type=str, default="datasets_prep/bm/step3", help="dataset origin (preprocessed)") 
    parser.add_argument("--src_npz", type=str, default="datasets/bm/step3", help="dataset origin (npz format)") 
    parser.add_argument("--model_v", type=str, default="runs/bm/step3_part1", help="model to predict v(tend) - needed for proper renorming during inference") 
    parser.add_argument("--model_T", type=str, default="runs/bm/step3_part3", help="model to predict T - needed for proper renorming during inference") 
    parser.add_argument("--replace_v", action="store_true", help="replace v at index 3 and 4 or default: append new wf-inputs to the end of the inputs")

    args = parser.parse_args()
    # assert args.step in ["3", "3b"], f"step {args.step} not tested"
    args.src_pt = Path(args.src_pt)
    args.src_npz = Path(args.src_npz)
    args.model_v = Path(args.model_v)
    args.model_T = Path(args.model_T)

    main_infer(args.src_pt, args.model_v, args.model_T, args.replace_v, args.src_npz)