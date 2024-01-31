import shutil

from utils.utils_data import SettingsTraining
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_paths import Paths1HP, set_paths_1hpnn

def prepare_data_and_paths(settings:SettingsTraining, dataset_raw:str):
    paths: Paths1HP
    settings.dataset_prep = "" # TODO rm
    paths, destination_dir = set_paths_1hpnn(dataset_raw, settings.inputs, settings.dataset_prep, problem=settings.problem) 
    settings.dataset_prep = paths.dataset_1st_prep_path
    settings.make_destination_path(destination_dir)
    settings.save_notes()
    settings.make_model_path(destination_dir)

    # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
    if not settings.dataset_prep.exists() or settings.case == "test":
        prepare_dataset_for_1st_stage(paths, settings)
    print(f"Dataset prepared ({paths.dataset_1st_prep_path})")


    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings