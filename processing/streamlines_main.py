from shutil import copytree
import time

from processing.networks.unetVariants import UNetNoPad2
from processing.streamlines_helpers import *

def build_streamlines(model_name:str, dataset_name:str="dataset_giant_100hp_varyK", based_on_pred:bool=True):
    ## copy ixydk files (later overwrite xyd)
    if based_on_pred:
        model_path = pathlib.Path(f"/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/{model_name}")
    data_dir = pathlib.Path("/scratch/sgs/pelzerja/datasets_prepared/allin1")
    origin_data_T_inout = "inputs_ixydk outputs_t"
    origin_data_T = data_dir/f"{dataset_name} {origin_data_T_inout}"
    origin_data_v = data_dir/f"{dataset_name} inputs_pki outputs_xy"
    origin_data_T_inout = "inputs_ixydk+s_outer outputs_t"
    destination = data_dir/f"{dataset_name} {origin_data_T_inout}" #/ds statt dataset_name
    if based_on_pred:
        destination += f" prep_with_{model_path.name}"
    copytree(origin_data_T,destination, dirs_exist_ok=True)

    idx_vx = 1
    idx_vy = 2
    idx_sf = 3
    idx_sf_outers = 5

    if based_on_pred:
        # renew args
        args = load_yaml(destination / "args.yaml")
        args["inputs"][idx_vx] = f"Liquid X-Velocity [m_per_y] - predicted by '{model_name}'"
        args["inputs"][idx_vy] = f"Liquid Y-Velocity [m_per_y] - predicted by '{model_name}'"
        save_yaml(args, destination / "args.yaml")
    # TODO manually: add s_outer to args, info

    if based_on_pred:
        # load model
        info_model = load_yaml(model_path / "info.yaml")
        settings_model = load_yaml(model_path / "settings.yaml")
        model = UNetNoPad2(in_channels=len(settings_model["inputs"]), out_channels=2, kernel_size=settings_model["kernel_size"], depth=settings_model["depth"], init_features=settings_model["init_features"])
        model.load(model_path)
        model.eval()

    for runid in (destination / "Inputs").iterdir():
        start_time = time.time()
        if based_on_pred:
            runid = runid.name
            data_in_model = torch.load(origin_data_v / "Inputs" /runid)
            vv_out = model(data_in_model.unsqueeze(0)).detach().squeeze(0)
            print(vv_out.shape)
            norm_model = NormalizeTransform(info_model)
            norm_model.reverse(vv_out, "Labels")
        inputs = torch.load(destination/"Inputs"/runid)
        norm = NormalizeTransform(load_yaml(destination/"info.yaml"))
        norm.reverse(inputs, "Inputs")

        if based_on_pred:
            # crop inputs to match model output
            required_size = vv_out.shape[1:]
            start_pos = ((inputs.shape[1] - required_size[0])//2, (inputs.shape[2] - required_size[1])//2)
            inputs_reduced = inputs[:, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

        if based_on_pred:
            # overwrite vx, vy with model output
            inputs_reduced[idx_vx] = vv_out[0]
            inputs_reduced[idx_vy] = vv_out[1]
        else:
            inputs_reduced = inputs
        inputs_reduced = inputs_reduced.numpy()

        # make streamlines
        _, streamlines_faded = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx_vx], vy=inputs_reduced[idx_vy], dims=inputs_reduced[0].shape)

        _, streamlines_faded_top = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx_vx], vy=inputs_reduced[idx_vy], dims=inputs_reduced[0].shape, offset=10)
        _, streamlines_faded_bottom = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx_vx], vy=inputs_reduced[idx_vy], dims=inputs_reduced[0].shape, offset=-10)

        # norm inputs acc. to info
        inputs_normed = norm(torch.tensor(inputs_reduced), "Inputs")
        inputs_normed[idx_sf] = streamlines_faded.unsqueeze(0)
        inputs_normed = torch.cat((inputs_normed, torch.zeros_like(inputs_normed[:1])), dim=0)
        inputs_normed[idx_sf_outers] = streamlines_faded_top.unsqueeze(0) + streamlines_faded_bottom.unsqueeze(0)
        print("new: ", inputs_normed.shape)

        save_new_datapoint(destination, runid, inputs_normed)
        print(f"Finished {runid} after {time.time()-start_time} seconds")

if __name__ == "__main__":
    dataset_name = "giant_double_100hp_v2"
    model_name = "paper24 finals/BEST_predict_v_v4"
    based_on_predicted_v = False
    build_streamlines(model_name, dataset_name, based_on_predicted_v)