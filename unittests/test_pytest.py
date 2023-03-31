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

def test_compute_data_max_and_min():
    # Fixture
    data = np.array([[[[1.0,2.0,3],[4,5,6]]]])
    # Expected result
    max = np.array([[[[6]]]])
    min = np.array([[[[1]]]])
    # Actual result
    actual_result = trans._compute_data_max_and_min(data)
    # Test
    assert actual_result == (max, min)
    # does not test keepdim part - necessary?
    # float numbers? expected_vlaue = pytest.approx(value, abs=0.001)

def test_sdf_transform():
    # Fixture
    testor = Tensor(np.array([[[1,2,3],[100,5,6]]]))
    data = utils.PhysicalVariables(time="now", properties=["Material ID"])
    data["Material ID"] = testor
    # Expected result
    sdf = Tensor(np.array([[[1,1.4142,2.2361],[0,1,2]]]))
    # Actual result
    sdf_transform = trans.SignedDistanceTransform()
    actual_result = sdf_transform(data)
    # Test
    assert eq(actual_result["Material ID"].value, sdf).flatten

def test_normalize_transform():
    data = utils.PhysicalVariables(time="now", properties=["test"])
    data["test"] = Tensor(np.array([[[[-2, -1, -1],[-1, 0, -1], [-1, -1, -1]]]]))
    mean_val = {"test": -1}
    std_val = {"test": 0.5}
    data_norm = Tensor(np.array([[[[-2, 0, 0],[0, 2, 0], [0, 0, 0]]]]))
    transform = trans.NormalizeTransform()
    tensor_eq = eq(transform(data, mean_val, std_val)["test"].value, data_norm).flatten
    assert tensor_eq

def test_reduceto2d_transform():
    data = utils.PhysicalVariables(time="now", properties=["test"])
    data["test"] = zeros([4,2,4])
    data_reduced = zeros([1,2,4])
    # Actual result
    data_actual = trans.ReduceTo2DTransform()(data, loc_hp=[0,1,1])
    # Test
    assert data_actual["test"].value.shape == data_reduced.shape
    tensor_eq = eq(data_actual["test"].value, data_reduced).flatten
    assert tensor_eq

def test_separate_property_unit():
    # Fixture
    fixture = "Liquid X-Velocity [m_per_y]"
    # Expected result
    name_expected = "Liquid X-Velocity"
    unit_expected = "m_per_y"
    # Actual result
    name, unit = utils.separate_property_unit(fixture)
    # Test
    assert name == name_expected
    assert unit == unit_expected

    # Fixture
    fixture = "Liquid X-Velocity"
    # Expected result
    name_expected = "Liquid X-Velocity"
    unit_expected = None
    # Actual result
    name, unit = utils.separate_property_unit(fixture)
    # Test
    assert name == name_expected
    assert unit == unit_expected

    # Fixture
    fixture = " [m_per_y]"
    # Expected result
    name_expected = ""
    unit_expected = "m_per_y"
    # Actual result
    name, unit = utils.separate_property_unit(fixture)
    # Test
    assert name == name_expected
    assert unit == unit_expected

def test_physical_property():
    time = "now [s]"
    expected_temperature = utils.PhysicalVariable("Temperature [K]")
    properties = ["Temperature [K]", "Pressure [Pa]"]
    physical_properties = utils.PhysicalVariables(time, properties)
    physical_properties["Temperature [K]"]=Tensor([3])
    physical_properties["Pressure [Pa]"]=2
    physical_properties["ID [-]"]=0

    assert physical_properties["Temperature [K]"].__repr__()=="Temperature (in K) with (1,) elements", "repr not working"
    assert physical_properties["Pressure [Pa]"].__repr__()=="Pressure (in Pa) with 1 elements", "repr not working"
    assert physical_properties.get_names_without_unit()==["Temperature", "Pressure", "ID"], "get_names_without_unit() not working"
    assert physical_properties["Temperature [K]"].value == 3, "value not set correctly"
    assert len(physical_properties)==3, "len not working"
    assert physical_properties["Temperature [K]"].unit == "K", "unit not set correctly"
    assert physical_properties["ID [-]"].value == 0, "value not set correctly"
    assert physical_properties["ID [-]"].unit == "-", "unit not set correctly"
    assert physical_properties["ID [-]"].__repr__()=="ID (in -) with 1 elements", "repr not working"
    assert physical_properties["ID [-]"].name_without_unit == "ID", "name_without_unit not set correctly"
    assert physical_properties["ID [-]"].id_name == "ID [-]", "id_name not set correctly"
    assert physical_properties.get_ids_list()==["Temperature [K]", "Pressure [Pa]", "ID [-]"], "get_ids not working"
    assert list(physical_properties.keys()) == ["Temperature [K]", "Pressure [Pa]", "ID [-]"], "keys not working"
    # test PhysicalVariable.__eq__()
    assert physical_properties["Temperature [K]"] != expected_temperature, "PhysicalVariable.__eq__() failed"
    expected_temperature.value = 3
    assert expected_temperature.value == 3, "value not set correctly"
    assert physical_properties["Temperature [K]"] == expected_temperature, "PhysicalVariable.__eq__() failed"

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

def test_datapoint_to_tensor_with_channel():
    inputs = utils.PhysicalVariables("now", properties=["temperature", "id"])
    labels = utils.PhysicalVariables("later", properties=["velocity"])
    inputs["temperature"] = zeros(2,3,4)
    inputs["id"] = zeros(2,3,4)
    labels["velocity"] = zeros(2,3,4)
    assert inputs["id"].shape() == (2,3,4), "shape not set correctly"
    datapoint = utils.DataPoint(0, inputs=inputs, labels=labels)
    # datapoint2 = utils.DataPoint(1, inputs=inputs, labels=labels)
    batch = dataloader._datapoint_to_tensor_including_channel(datapoint)
    assert batch.inputs.shape == (2,2,3,4), "_datapoint_to_tensor_with_channel does not concat the channels correctly"

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

# def test_reverse_transform():
#     _, dataloaders_2D = lp.init_data(dataset_name="test_dataset_bm_10", #"OLD_bash_file_and_script_structure/groundtruth_hps_no_hps/groundtruth_hps_overfit_10", 
#         reduce_to_2D=True, reduce_to_2D_xy=True,
#         inputs="pk", labels="t") #, batch_size=3)

#     for _, data in enumerate(dataloaders_2D["train"]):
#         for data in dataloaders_2D["train"].dataset:
#             dataloaders_2D["train"].dataset.reverse_transform(data)

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


if __name__ == "__main__":
    test_data_init()
    # test_combinations()
    # test_dataloader_iter()
    # test_normalize_transform()
    # test_sdf_transform()