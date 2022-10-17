import data.transforms as trans    # The code to test
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
    actual_result = trans.compute_data_max_and_min(data)
    # Test
    assert actual_result == (max, min)
    # does not test keepdim part - necessary?
    # float numbers? expected_vlaue = pytest.approx(value, abs=0.001)


def test_normalize_transform():
    data = torch.Tensor(np.array([[[[-2, -1, -1],[-1, 0, -1], [-1, -1, -1]]]]))
    data_norm = torch.Tensor(np.array([[[[-2, 0, 0],[0, 2, 0], [0, 0, 0]]]]))
    transform = trans.NormalizeTransform()
    tensor_eq = torch.eq(transform(data), data_norm).flatten
    assert tensor_eq

def test_data_init():
    _, dataloaders = lp.init_data(reduce_to_2D=True, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_1", inputs="xyz")
    assert dataloaders["train"].dataset[0]['x'].shape == (4,128,16)
    assert len(dataloaders) == 3
    _, dataloaders = lp.init_data(reduce_to_2D=True, overfit=True, dataset_name="groundtruth_hps_no_hps/groundtruth_hps_overfit_1", inputs="xysd")
    assert dataloaders["train"].dataset[0]['x'].shape == (3,128,16)

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
    properties = {"temperature": utils.PhysicalVariable("Temperature [K]"),
        "pressure": utils.PhysicalVariable("Pressure [Pa]")}
    physical_properties = utils.PhysicalVariables(time, properties)
    physical_properties["pressure"]=2
    physical_properties["temperature"]=3
    physical_properties["ID [-]"]=0

    assert physical_properties["temperature"].__repr__()=="Temperature (in K)", "repr not working"
    assert physical_properties["pressure"].__repr__()=="Pressure (in Pa)", "repr not working"
    assert physical_properties.get_names_without_unit()==["Temperature", "Pressure", "ID"], "get_names_without_unit() not working"
    assert physical_properties["temperature"].value == 3, "value not set correctly"
    assert len(physical_properties)==3, "len not working"
    assert physical_properties["temperature"].unit == "K", "unit not set correctly"
    assert physical_properties["ID [-]"].value == 0, "value not set correctly"
    assert physical_properties["ID [-]"].unit == "-", "unit not set correctly"
    assert physical_properties["ID [-]"].__repr__()=="ID (in -)", "repr not working"
    assert physical_properties["ID [-]"].name_without_unit == "ID", "name_without_unit not set correctly"
    assert physical_properties["ID [-]"].id_name == "ID [-]", "id_name not set correctly"
    assert physical_properties.get_ids()==["Temperature [K]", "Pressure [Pa]", "ID [-]"], "get_ids not working"
    assert list(physical_properties.keys()) == ['temperature', 'pressure', 'ID [-]'], "keys not working"
    # test PhysicalVariable.__eq__()
    assert physical_properties["temperature"] != expected_temperature, "PhysicalVariable.__eq__() failed"
    expected_temperature.value = 3
    assert expected_temperature.value == 3, "value not set correctly"
    assert physical_properties["temperature"] == expected_temperature, "PhysicalVariable.__eq__() failed"
    
