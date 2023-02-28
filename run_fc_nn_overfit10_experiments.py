from main import run_experiment
from data.utils import load_settings
# import tensorboard

kwargs = {}
kwargs = load_settings(".", "settings_training")
kwargs["model_choice"] = "fc"
kwargs["lr"]=1e-4#7
kwargs["overfit"] = True
kwargs["dataset_name"] = "dataset_pressure_vary_perm_iso_3D_10dp"
input_combis = ["p", "t", "", "px", "py", "xy", "pt"]
kwargs["n_epochs"] = 10000
for input in input_combis:
    kwargs["inputs"] = input
    kwargs["name_folder_destination"] = f"{kwargs['model_choice']}_overfit_{kwargs['overfit']}_epochs_{kwargs['n_epochs']}_inputs_{kwargs['inputs']}_perm_iso_10dp"
    run_experiment(**kwargs) #, normalize=False)