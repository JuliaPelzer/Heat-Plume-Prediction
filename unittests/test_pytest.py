import data.transforms as trans    # The code to test
import learn_process as lp

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
