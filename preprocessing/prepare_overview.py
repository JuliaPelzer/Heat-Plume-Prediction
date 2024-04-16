import shutil
from typing import Union

from torch import Tensor

from preprocessing.prepare_1hp_boxes import prepare_dataset_for_1st_stage
from preprocessing.prepare_2ndstage import prepare_dataset_for_2nd_stage
from preprocessing.prepare_paths import (Paths1HP, Paths2HP, set_paths_1hpnn,
                                         set_paths_2hpnn)
from utils.utils_data import SettingsTraining


def prepare_paths_and_settings(settings:SettingsTraining):
    if not settings.case_2hp:
        paths: Paths1HP
        paths, destination_dir = set_paths_1hpnn(settings.dataset_raw, settings.inputs, settings.dataset_prep, problem=settings.problem) 
        settings.dataset_prep = paths.dataset_1st_prep_path
        inputs_2hp = None
    else:
        paths: Paths2HP
        inputs_2hp = settings.inputs
        paths, settings.inputs, destination_dir = set_paths_2hpnn(settings.dataset_raw, settings.inputs, dataset_prep = settings.dataset_prep,)
        settings.dataset_prep = paths.datasets_boxes_prep_path

    settings.make_destination_path(destination_dir)
    settings.save_notes()
    settings.make_model_path(destination_dir)

    return paths, settings, inputs_2hp


def prepare_data(settings:SettingsTraining, paths: Union[Paths1HP, Paths2HP], inputs_2hp: str, additional_input: Tensor = None):
    if not settings.case_2hp:
        # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
        print(f"Preparing dataset ({paths.dataset_1st_prep_path}")
        if not paths.dataset_1st_prep_path.exists(): # or settings.case == "test": # breaks if case == "test" and dataset_prep exists because tries to write where its already written ?!
            prepare_dataset_for_1st_stage(paths, settings, additional_input=additional_input)
        print(f"Dataset prepared ({paths.dataset_1st_prep_path})")
    else:
        if not paths.dataset_1st_prep_path.exists() or not paths.dataset_1st_prep_path: # TODO settings.case == "test"?
            prepare_dataset_for_2nd_stage(paths, settings)
        settings.inputs = inputs_2hp
        print(f"Dataset prepared ({paths.datasets_boxes_prep_path})")

    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings

def prepare_data_and_paths(settings:SettingsTraining, additional_input: Tensor = None):
    paths, settings, inputs_2hp = prepare_paths_and_settings(settings)
    prepare_data(settings, paths, inputs_2hp, additional_input=additional_input)
    return settings