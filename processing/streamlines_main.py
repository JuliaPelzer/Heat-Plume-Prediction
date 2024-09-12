from shutil import copytree
import time

from processing.networks.unetVariants import UNetNoPad2
from processing.streamlines_helpers import *

def build_streamlines(model_name:str, dataset_name:str="dataset_giant_100hp_varyK", based_on_pred:bool=True):
    ## copy ixydk files (later overwrite xyd)
    if based_on_pred:
        model_path = pathlib.Path(f"/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/{model_name}")
    data_dir = pathlib.Path("/scratch/sgs/pelzerja/datasets_prepared/allin1")
    origin_data_T_inout = "inputs_ixyk outputs_t for_s" # base, replace xy, add s,s_outer
    origin_data_T = data_dir/f"{dataset_name} {origin_data_T_inout}"
    origin_data_v = data_dir/f"{dataset_name} inputs_pki outputs_xy"
    destination_data_T_inout = "inputs_ixydk+s_outer outputs_t"
    if based_on_pred:
            destination = data_dir/f"{dataset_name} {destination_data_T_inout} prep_with_{model_path.name}"
    else:
        destination = data_dir/f"{dataset_name} {destination_data_T_inout}" #/ds statt dataset_name
    copytree(origin_data_T,destination, dirs_exist_ok=True)

    idx = {"vx": 1,
            "vy" : 2,
            "sf" : 3,
            "sf_outers" : 5}

    if based_on_pred:
        # load model
        info_v = load_yaml(model_path / "info.yaml")
        settings_model = load_yaml(model_path / "settings.yaml")
        model = UNetNoPad2(in_channels=len(settings_model["inputs"]), out_channels=2, kernel_size=settings_model["kernel_size"], depth=settings_model["depth"], init_features=settings_model["init_features"])
        model.load(model_path)
        model.eval()
    else:
        info_v = load_yaml(origin_data_v / "info.yaml")
    norm_v = NormalizeTransform(info_v)

    norm_before = NormalizeTransform(load_yaml(destination/"info.yaml"))
    if based_on_pred:
        v_info_path = model_path
    else:
        v_info_path = origin_data_v
    correct_args_info(destination, v_info_path, based_on_pred)
    norm_after = NormalizeTransform(load_yaml(destination/"info.yaml"))

    for runid in (destination / "Inputs").iterdir():
        runid = runid.name
        start_time = time.time()
        if based_on_pred:
            data_in_model = torch.load(origin_data_v/"Inputs"/runid)
            vv = model(data_in_model.unsqueeze(0)).detach().squeeze(0)
        else:
            vv = torch.load(origin_data_v/"Labels"/runid)
        norm_v.reverse(vv, "Labels")
        inputs = torch.load(destination/"Inputs"/runid)
        norm_before.reverse(inputs, "Inputs")

        inputs = extend_inputs_dims(inputs)
        # crop inputs to match model output
        required_size = vv.shape[1:]
        start_pos = ((inputs.shape[1] - required_size[0])//2, (inputs.shape[2] - required_size[1])//2)
        inputs_reduced = inputs[:, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

        # overwrite vx, vy with model output
        inputs_reduced[idx["vx"]] = vv[0]
        inputs_reduced[idx["vy"]] = vv[1]
        inputs_reduced = inputs_reduced.numpy()

        # make streamlines
        _, streamlines_faded = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape)

        _, streamlines_faded_top = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, offset=10)
        _, streamlines_faded_bottom = make_streamlines(mat_ids=inputs_reduced[0], vx=inputs_reduced[idx["vx"]], vy=inputs_reduced[idx["vy"]], dims=inputs_reduced[0].shape, offset=-10)

        # norm inputs acc. to info
        inputs_normed = norm_after(torch.tensor(inputs_reduced), "Inputs")
        inputs_normed[idx["sf"]] = streamlines_faded.unsqueeze(0)
        inputs_normed[idx["sf_outers"]] = streamlines_faded_top.unsqueeze(0) + streamlines_faded_bottom.unsqueeze(0)
        print("new: ", inputs_normed.shape)

        save_new_datapoint(destination, runid, inputs_normed)
        print(f"Finished {runid} after {time.time()-start_time} seconds")


if __name__ == "__main__":
    dataset_name = "dataset_giant_100hp_varyK"
    model_name = "paper24 finals/BEST_predict_v_v4"
    based_on_predicted_v = True
    build_streamlines(model_name, dataset_name, based_on_predicted_v)