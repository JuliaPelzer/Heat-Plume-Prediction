import pytest
import os
import data.transforms as trans    # The code to test
import data.dataloader as dataloader
import data.dataset_loading as lp
import data.utils as utils
import networks.losses

# additional libraries
import numpy as np
from torch import Tensor, zeros, eq, Size, mean, std

def test_sdf_transform():
    # TODO
    pass
    # # Fixture
    # testor = Tensor(np.array([[[1,2,3],[100,5,6]]]))
    # data = utils.PhysicalVariables(time="now", properties=["Material ID"])
    # data["Material ID"] = testor
    # # Expected result
    # sdf = Tensor(np.array([[[1,1.4142,2.2361],[0,1,2]]]))
    # # Actual result
    # sdf_transform = trans.SignedDistanceTransform()
    # actual_result = sdf_transform(data)
    # # Test
    # assert eq(actual_result["Material ID"].value, sdf).flatten

def test_normalize_transform():
    # TODO
    pass
    # data = utils.PhysicalVariables(time="now", properties=["test"])
    # data["test"] = Tensor(np.array([[[[-2, -1, -1],[-1, 0, -1], [-1, -1, -1]]]]))
    # mean_val = {"test": -1}
    # std_val = {"test": 0.5}
    # data_norm = Tensor(np.array([[[[-2, 0, 0],[0, 2, 0], [0, 0, 0]]]]))
    # transform = trans.NormalizeTransform()
    # tensor_eq = eq(transform(data, mean_val, std_val)["test"].value, data_norm).flatten
    # assert tensor_eq

def test_reduceto2d_transform():
    # TODO
    pass
    # data = utils.PhysicalVariables(time="now", properties=["test"])
    # data["test"] = zeros([4,2,4])
    # data_reduced = zeros([1,2,4])
    # # Actual result
    # data_actual = trans.ReduceTo2DTransform()(data, loc_hp=[0,1,1])
    # # Test
    # assert data_actual["test"].value.shape == data_reduced.shape
    # tensor_eq = eq(data_actual["test"].value, data_reduced).flatten
    # assert tensor_eq

# def test_data_init():
#     name_folder_destination = "test_bm"
#     try:
#         os.mkdir(os.path.join(os.getcwd(), "runs", name_folder_destination))
#     except:
#         pass

#     # _, dataloaders = lp.init_data(reduce_to_2D=False, dataset_name="test_dataset_01", inputs="xyz", name_folder_destination=name_folder_destination)
#     # assert len(dataloaders) == 2
#     # assert len(dataloaders["train"].dataset[0].inputs) == 4
#     # for prop in dataloaders["train"].dataset[0].inputs.values():
#     #     assert prop.shape() == (8,128,8)
#     #     break

#     _, dataloaders = lp.init_data(reduce_to_2D=True, dataset_name="test_dataset_bm_10", inputs="pk", name_folder_destination=name_folder_destination)
#     assert len(dataloaders) == 2
#     assert len(dataloaders["train"].dataset[0].inputs) == 3
#     for prop in dataloaders["train"].dataset[0].inputs.values():
#         assert prop.shape() == (8,128,8)
#         break

#     _, dataloaders = lp.init_data(reduce_to_2D=True, dataset_name="test_dataset_bm_01", inputs="pk", name_folder_destination=name_folder_destination)
#     for prop in dataloaders["train"].dataset[0].inputs.values():
#         assert prop.shape() == (128,32)
#         break
    
#     _, dataloaders = lp.init_data(reduce_to_2D=False, dataset_name="test_dataset_bm_01", inputs="p", name_folder_destination=name_folder_destination)
#     assert len(dataloaders["train"].dataset[0].inputs) == 2
#     for prop in dataloaders["train"].dataset[0].inputs.values():
#         assert prop.shape() == (8,128,8)
#         break
    
#     _, dataloaders = lp.init_data(reduce_to_2D=False, dataset_name="test_dataset_bm_01", inputs="", name_folder_destination=name_folder_destination)
#     assert len(dataloaders["train"].dataset[0].inputs) == 1
#     assert dataloaders["train"].dataset[0].inputs.get_ids_list() == ["Material ID"] or dataloaders["train"].dataset[0].inputs.get_ids_list() == ["Material ID"]
#     for prop in dataloaders["train"].dataset[0].inputs.values():
#         assert prop.shape() == (8,128,8)
#         break

#     path = os.path.join("runs", name_folder_destination)
#     os.system(f"rm -r {path}")

# def test_combinations():
#     datasets, dataloaders = lp.init_data(dataset_name="test_dataset_bm_10", inputs="xyz")
#     assert len(datasets["train"])+len(datasets["val"])+len(datasets["test"])==10, "number of runs in datasets should be 10"
#     assert dataloaders["train"].dataset[0], "getitem/ load datapoint does not really work in combination with init_data"
#     assert datasets["train"][0].inputs["Liquid X-Velocity [m_per_y]"].__repr__() == "Liquid X-Velocity (in m_per_y) with (128, 8) elements", "combination of data_init and repr of PhysicalVariable does not work"
#     datasets, dataloaders = lp.init_data(reduce_to_2D=False, dataset_name="test_dataset_bm_10", inputs="xyz")
#     assert datasets["train"][0].inputs["Liquid X-Velocity [m_per_y]"].__repr__() == "Liquid X-Velocity (in m_per_y) with (8, 128, 8) elements", "combination of data_init and repr of PhysicalVariable does not work"
#     assert not datasets["train"][0].inputs["Liquid X-Velocity [m_per_y]"] == datasets["train"][1].inputs["Liquid X-Velocity [m_per_y]"], "same values in two datapoints?! that should not happen - problem with copying"

# def test_dataloader_iter():
#     name_folder_destination = "test_bm"
#     try:
#         os.mkdir(os.path.join(os.getcwd(), "runs", name_folder_destination))
#     except:
#         pass

#     _, dataloaders = lp.init_data(dataset_name="test_dataset_bm_10", reduce_to_2D=True, batch_size=4, name_folder_destination=name_folder_destination)
#     sizes_x = []
#     sizes_y = []
#     for dataloader in dataloaders.values():
#         for _, datapoint in enumerate(dataloader):
#             x = datapoint.inputs.float()
#             y = datapoint.labels.float()
#             sizes_x.append(Tensor.size(x))
#             sizes_y.append(Tensor.size(y))
#     assert(sizes_x == [Size([4, 6, 8, 128, 8]), Size([3, 6, 8, 128, 8]), Size([2, 6, 8, 128, 8]), Size([1, 6, 8, 128, 8])]), "enumerate(dataloader) does not work properly"
#     assert(sizes_y == [Size([4, 4, 8, 128, 8]), Size([3, 4, 8, 128, 8]), Size([2, 4, 8, 128, 8]), Size([1, 4, 8, 128, 8])]), "enumerate(dataloader) does not work properly"
#     # first number depends on split + batch size ´-> check whether split changed

def test_visualize_data():
    pass

def test_train_model():
    pass
    """ Test input format etc. of train_model"""

def test_mselossexcludenotchangedtemp():
    import torch
    # Fixture
    tensor1 = torch.tensor([[10.6, 12], [12, 10.6]])
    tensor2 = torch.tensor([[10.6, 12], [10.6, 12]])
    # Expected result
    expected_value = torch.tensor(1.30666667)
    # Actual result
    value = networks.losses.MSELossExcludeNotChangedTemp(ignore_temp=10.6)(tensor1, tensor2)
    # Test
    assert expected_value == pytest.approx(value)


# if __name__ == "__main__":
#     test_data_init()