import torch
from domain_classes.domain import Domain

from preprocessing.prepare_2ndstage import main_merge_inputs
from utils.prepare_paths import set_paths_2hpnn

# def test_norm():
# TODO methodology is old
#     (
#         datasets_raw_domain_dir,
#         datasets_prepared_domain_dir,
#         dataset_domain_path,
#         datasets_model_trained_with_path,
#         model_path,
#         name_extension,
#         datasets_prepared_2hp_dir,
#     ) = set_paths_2hpnn(
#         "dataset_2hps_1fixed_testing", # does not exist anymore
#         "current_unet_benchmark_dataset_2d_100datapoints_input_empty_T_0",
#         "benchmark_dataset_2d_100datapoints_input_empty_T_0",
#         "g",
#     )
#     dummy_domain = Domain(dataset_domain_path, stitching_method="max")

#     test_value = [1, 3.5, 0.27]
#     test_value = torch.tensor(test_value)
#     result = dummy_domain.reverse_norm(dummy_domain.norm(test_value))

#     assert torch.allclose(result, test_value)

def test_merged_inputs():
    main_merge_inputs("dataset_2hps_1fixed_10dp inputs_gki100 boxes", True)

