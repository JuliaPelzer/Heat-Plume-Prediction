import shutil

from processing.training import train

def test_main_1ststage():
    args = {
        "data_raw" :         "dataset_2d_small_1000dp",
        "data_prep" :        "",
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
    train(args)
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
        "data_raw" :         "dataset_giant_100hp_varyK",
        "data_prep" :        "",
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
    train(args)
    shutil.rmtree("runs/allin1/unittest", ignore_errors=True)


def test_main_extend():
    args = {
        "data_raw" :         "dataset_medium-10dp",
        "data_prep" :        "",
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
    train(args)
    shutil.rmtree("runs/extend/unittest", ignore_errors=True)