import data.transforms as trans    # The code to test
import data.dataloader as dataloader
import learn_process as lp
import data.utils as utils

# additional libraries
import numpy as np
import torch

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


def test_normalize_transform():
    data = utils.PhysicalVariables(time="now", properties=["test"])
    data["test"] = torch.Tensor(np.array([[[[-2, -1, -1],[-1, 0, -1], [-1, -1, -1]]]]))
    data_norm = torch.Tensor(np.array([[[[-2, 0, 0],[0, 2, 0], [0, 0, 0]]]]))
    transform = trans.NormalizeTransform()
    tensor_eq = torch.eq(transform(data)["test"].value, data_norm).flatten
    assert tensor_eq

def test_reduceto2d_transform():
    data = utils.PhysicalVariables(time="now", properties=["test"])
    data["test"] = torch.zeros([4,2,4])
    data_reduced = torch.zeros([1,2,4])
    # Actual result
    data_actual = trans.ReduceTo2DTransform(x=0)(data)
    # Test
    assert data_actual["test"].value.shape == data_reduced.shape
    tensor_eq = torch.eq(data_actual["test"].value, data_reduced).flatten
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
    physical_properties["Temperature [K]"]=torch.Tensor([3])
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

def test_data_init():
    _, dataloaders = lp.init_data(reduce_to_2D=False, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_01", inputs="xyz")
    assert len(dataloaders) == 3
    assert len(dataloaders["train"].dataset[0].inputs) == 4
    for prop in dataloaders["train"].dataset[0].inputs.values():
        assert prop.shape() == (16,128,16)
        break

    _, dataloaders = lp.init_data(reduce_to_2D=True, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_01", inputs="xyz")
    for prop in dataloaders["train"].dataset[0].inputs.values():
        assert prop.shape() == (128,16)
        break
    
    _, dataloaders = lp.init_data(reduce_to_2D=False, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_01", inputs="xyzp")
    assert len(dataloaders["train"].dataset[0].inputs) == 5
    for prop in dataloaders["train"].dataset[0].inputs.values():
        assert prop.shape() == (16,128,16)
        break
    
    _, dataloaders = lp.init_data(reduce_to_2D=False, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_01", inputs="")
    assert len(dataloaders["train"].dataset[0].inputs) == 1
    assert dataloaders["train"].dataset[0].inputs.get_ids_list() == ["Material_ID"]
    for prop in dataloaders["train"].dataset[0].inputs.values():
        assert prop.shape() == (16,128,16)
        break

def test_combinations():
    datasets, dataloaders = lp.init_data(dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_10", inputs="xyz")
    assert len(datasets["train"])+len(datasets["val"])+len(datasets["test"])==10, "number of runs in datasets should be 10"
    assert dataloaders["train"].dataset[0], "getitem/ load datapoint does not really work in combination with init_data"
    assert datasets["train"][0].inputs["Liquid X-Velocity [m_per_y]"].__repr__() == "Liquid X-Velocity (in m_per_y) with (128, 16) elements", "combination of data_init and repr of PhysicalVariable does not work"
    datasets, dataloaders = lp.init_data(reduce_to_2D=False, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_10", inputs="xyz")
    assert datasets["train"][0].inputs["Liquid X-Velocity [m_per_y]"].__repr__() == "Liquid X-Velocity (in m_per_y) with (16, 128, 16) elements", "combination of data_init and repr of PhysicalVariable does not work"
    assert not datasets["train"][0].inputs["Liquid X-Velocity [m_per_y]"] == datasets["train"][1].inputs["Liquid X-Velocity [m_per_y]"], "same values in two datapoints?! that should not happen - problem with copying"

def test_datapoint_to_tensor_with_channel():
    inputs = utils.PhysicalVariables("now", properties=["temperature", "id"])
    labels = utils.PhysicalVariables("later", properties=["velocity"])
    inputs["temperature"] = torch.zeros(2,3,4)
    inputs["id"] = torch.zeros(2,3,4)
    labels["velocity"] = torch.zeros(2,3,4)
    assert inputs["id"].shape() == (2,3,4), "shape not set correctly"
    datapoint = utils.DataPoint(0, inputs=inputs, labels=labels)
    # datapoint2 = utils.DataPoint(1, inputs=inputs, labels=labels)
    batch = dataloader._datapoint_to_tensor_with_channel(datapoint)
    assert batch.inputs.shape == (2,2,3,4), "_datapoint_to_tensor_with_channel does not concat the channels correctly"

def test_visualize_data():
    pass

def test_train_model():
    pass
    """ Test input format etc. of train_model"""