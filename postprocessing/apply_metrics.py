from pathlib import Path

from processing.networks.unetVariants import UNetNoPad2
from processing.networks.unet import UNet
from preprocessing.data_init import init_data
from postprocessing.measurements import measure_losses_paper24
from utils.utils_args import save_yaml

if __name__=="__main__":
    # 1hp-cnn data_prep = Path("/scratch/sgs/pelzerja/datasets_prepared/1hp_boxes/dataset_2d_small_1000dp inputs_gksi")
    # 2hp-cnn
    data_prep = Path("/scratch/sgs/pelzerja/datasets_others/paper23_etc/2HP_boxes/dataset_2hps_1fixed_1000dp_2hp_gksi_1000dp")
    # allin1 data_prep = Path("/scratch/sgs/pelzerja/datasets_prepared/allin1/ds inputs_ixydk outputs_t prep_with_BEST_predict_v_v4")
    args = {"problem": "2stages", "case": "test", "data_prep": data_prep, 
            "len_box": 2560, "skip_per_dir": 32, "device":"cpu"}
    
    # 1hp-cnn model_path = Path("/scratch/sgs/pelzerja/models/paper23/best_models_1hpnn/gksi1000/current_unet_dataset_2d_small_1000dp_gksi_v7")
    # 2hp-cnn
    model_path = Path("/scratch/sgs/pelzerja/models/paper23/best_models_2hpnn/1000dp_1000gksi_separate/current_unet_dataset_2hps_1fixed_1000dp_2hp_gksi_1000dp_v1")
    # model_path = Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/predict_T_from_s hyperparam_opt/MAE_real_trial11 BEST_MAE")
    model_name = "model.pt"
    # 1hp-cnn, 2hp-cnn
    model = UNet(in_channels=2)
    # model = UNetNoPad2(in_channels=5, out_channels=1, kernel_size=4, depth=4, init_features=32)
    model.load(model_path, model_name=model_name, device=args["device"])

    input_channels, output_channels, dataloaders = init_data(args)

    metrics = measure_losses_paper24(model, dataloaders, args, vT_case = "temperature")
    save_yaml(metrics, model_path / "metrics.yaml")