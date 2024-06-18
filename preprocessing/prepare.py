import shutil

from data_stuff.utils import SettingsTraining
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_paths import Paths1HP,  set_paths_1hpnn

def prepare_data_and_paths(settings:SettingsTraining):
    if settings.case_2hp:
        assert False, "not implemented on this branch"
    else:
        paths: Paths1HP
        paths, destination_dir = set_paths_1hpnn(settings.dataset_raw, settings.inputs, settings.dataset_prep, problem=settings.problem, augmentation=settings.augmentation) 
        settings.dataset_prep = paths.dataset_1st_prep_path

        settings.make_destination_path(destination_dir)
        settings.save_notes()
        settings.make_model_path(destination_dir)

        # prepare dataset if not done yet OR if case=test do it anyways because of potentially different std,mean,... values than trained with
        if not settings.dataset_prep.exists() or settings.case == "test": # if test, always want to prepare because the normalization parameters have to match
            print(settings.dataset_prep)
            prepare_dataset_for_1st_stage(paths, settings)
        print(f"Dataset prepared ({paths.dataset_1st_prep_path})")


    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings