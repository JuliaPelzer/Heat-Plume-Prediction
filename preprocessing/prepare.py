import shutil

from data_stuff.utils import SettingsTraining
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_paths import Paths1HP, Paths2HP,  set_paths_1hpnn, set_paths_2hpnn
from preprocessing.prepare_xhp_dataset import prepare_xhp_dataset

def prepare_data_and_paths(settings:SettingsTraining):
    if not settings.case == "prep_xhp":
        paths: Paths1HP
        paths, destination_dir = set_paths_1hpnn(settings.dataset_raw, settings.inputs, settings.dataset_prep, problem=settings.problem)
    else:
        paths: Paths2HP
        inputs_2hp = settings.inputs
        paths, settings.inputs, destination_dir = set_paths_2hpnn(settings.dataset_raw, settings.inputs,model_name=settings.model, dataset_prep = settings.dataset_prep,)
        settings.dataset_prep = paths.datasets_boxes_prep_path
    settings.dataset_prep = paths.dataset_1st_prep_path

    settings.make_destination_path(destination_dir)
    settings.save_notes()
    settings.make_model_path(destination_dir)

    if settings.case == "prep_xhp":
        prepare_xhp_dataset(paths, settings)
        return settings

    # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
    if not settings.dataset_prep.exists() or settings.case == "test": # if test, always want to prepare because the normalization parameters have to match
        #TODO handle case for 2HP where no raw dataset exits
        #print("skip preparing raw dataset as current tests do not have raw dataset in prepare.py")
        if not settings.only_prep:
            prepare_dataset_for_1st_stage(paths, settings)
    print(f"Dataset prepared ({paths.dataset_1st_prep_path})")


    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings