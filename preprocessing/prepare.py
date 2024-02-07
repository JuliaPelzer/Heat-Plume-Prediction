
import shutil

from data_stuff.utils import SettingsTraining
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_2ndstage import prepare_dataset_for_2nd_stage
from preprocessing.prepare_paths import Paths1HP, Paths2HP, set_paths_1hpnn, set_paths_2hpnn

def prepare_data_and_paths(settings:SettingsTraining):
    if not settings.case_2hp:
        paths: Paths1HP
        paths, destination_dir = set_paths_1hpnn(settings.dataset_raw, settings.inputs, settings.dataset_prep, problem=settings.problem) 
        settings.dataset_prep = paths.dataset_1st_prep_path

    else:
        assert settings.problem == "2stages", "2nd stage is only possible with 2stages problem"
        paths: Paths2HP
        paths, inputs_1hp, destination_dir = set_paths_2hpnn(settings.dataset_raw, settings.inputs, dataset_prep = settings.dataset_prep,)
        settings.dataset_prep = paths.datasets_boxes_prep_path
    settings.make_destination_path(destination_dir)
    settings.save_notes()
    settings.make_model_path(destination_dir)

    if not settings.case_2hp:
        # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
        if not settings.dataset_prep.exists(): # or settings.case == "test":
            print(settings.dataset_prep)
            prepare_dataset_for_1st_stage(paths, settings)
        print(f"Dataset prepared ({paths.dataset_1st_prep_path})")

    else:
        if not settings.dataset_prep.exists() or not paths.dataset_1st_prep_path: # TODO settings.case == "test"?
            prepare_dataset_for_2nd_stage(paths, inputs_1hp, settings.device)
        print(f"Dataset prepared ({paths.datasets_boxes_prep_path})")

    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings