import shutil

from main import main
from preprocessing.prepare_2ndstage import main_merge_inputs

def test_merged_inputs():
    main_merge_inputs("dataset_2hps_1fixed_10dp inputs_gki100 boxes", True)    
    
def test_main_1ststage():
    args = {
        "dataset_raw" :         "dataset_2d_small_1000dp",
        "dataset_train" :       None,
        "dataset_val" :         None,
        "dataset_test" :        None,
        "dataset_prep" :        "",
        "device" :              "cuda:0",
        "epochs" :              1,
        "case" :                "train",
        "model" :               "default",
        "destination" :         "unittest",
        "inputs" :              "gksi",
        "case_2hp" :            False,
        "visualize" :           True,
        "save_inference" :      False,
        "problem" :             "2stages",
        "notes" :               "",
        "len_box" :             256,
        "skip_per_dir" :        0,
    }
    main(args)
    shutil.rmtree("runs/1hpnn/unittest", ignore_errors=True)


# def test_main_2ndstage():
#     args = {
#         "dataset_raw" :         "dataset_2hps_1fixed_10dp", # TODO
#         "dataset_train" :       None,
#         "dataset_val" :         None,
#         "dataset_test" :        None,
#         "dataset_prep" :        "",
#         "device" :              "cuda:0",
#         "epochs" :              1,
#         "case" :                "train",
#         "model" :               "default",
#         "destination" :         "tmp",
#         "inputs" :              "gksi100",
#         "case_2hp" :            True,
#         "visualize" :           True,
#         "save_inference" :      False,
#         "problem" :             "2stages",
#         "notes" :               "",
#         "len_box" :             256,
#         "skip_per_dir" :        0,
#     }
#     main(args)
#     os.rmdir("runs/2hpnn/tmp")


def test_main_allin1():
    args = {
        "dataset_raw" :         None,
        "dataset_train" :       "dataset_giant_100hp_varyK_train",
        "dataset_val" :         "dataset_giant_100hp_varyK_val",
        "dataset_test" :        "dataset_giant_100hp_varyK_test",
        "dataset_prep" :        "",
        "device" :              "cuda:0",
        "epochs" :              1,
        "case" :                "train",
        "model" :               "default",
        "destination" :         "unittest",
        "inputs" :              "gkn",
        "case_2hp" :            False,
        "visualize" :           True,
        "save_inference" :      False,
        "problem" :             "allin1",
        "notes" :               "",
        "len_box" :             256,
        "skip_per_dir" :        32,
    }
    main(args)
    shutil.rmtree("runs/allin1/unittest", ignore_errors=True)


def test_main_extend():
    args = {
        "dataset_raw" :         "dataset_medium-10dp",
        "dataset_train" :       None,
        "dataset_val" :         None,
        "dataset_test" :        None,
        "dataset_prep" :        "",
        "device" :              "cuda:0",
        "epochs" :              1,
        "case" :                "train",
        "model" :               "default",
        "destination" :         "unittest",
        "inputs" :              "gk",
        "case_2hp" :            False,
        "visualize" :           True,
        "save_inference" :      False,
        "problem" :             "extend",
        "notes" :               "",
        "len_box" :             256,
        "skip_per_dir" :        16,
    }
    main(args)
    shutil.rmtree("runs/extend_plumes/unittest", ignore_errors=True)