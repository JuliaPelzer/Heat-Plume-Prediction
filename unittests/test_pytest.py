import pytest
import torch

import networks.losses
from data.transforms import ComposeTransform, SignedDistanceTransform


def test_sdf_transform():
    trafo = ComposeTransform([SignedDistanceTransform()])
    # Fixture
    data = {"SDF": torch.zeros((9, 9, 2))}
    data["SDF"][1,1,0] = 1
    # Expected result
    expected = torch.Tensor([
        [[0.8579, 0.8259], [0.8995, 0.8579], [0.8579, 0.8259], [0.7753, 0.7538], [0.6822, 0.6667], [0.5856, 0.5736], [0.4875, 0.4778], [0.3887, 0.3805], [0.2893, 0.2823]],
        [[0.8995, 0.8579], [1.0000, 0.8995], [0.8995, 0.8579], [0.7990, 0.7753], [0.6985, 0.6822], [0.5980, 0.5856], [0.4975, 0.4875], [0.3970, 0.3887], [0.2965, 0.2893]],
        [[0.8579, 0.8259], [0.8995, 0.8579], [0.8579, 0.8259], [0.7753, 0.7538], [0.6822, 0.6667], [0.5856, 0.5736], [0.4875, 0.4778], [0.3887, 0.3805], [0.2893, 0.2823]],
        [[0.7753, 0.7538], [0.7990, 0.7753], [0.7753, 0.7538], [0.7157, 0.6985], [0.6376, 0.6239], [0.5505, 0.5394], [0.4588, 0.4495], [0.3644, 0.3565], [0.2683, 0.2615]],
        [[0.6822, 0.6667], [0.6985, 0.6822], [0.6822, 0.6667], [0.6376, 0.6239], [0.5736, 0.5619], [0.4975, 0.4875], [0.4140, 0.4054], [0.3258, 0.3184], [0.2346, 0.2280]],
        [[0.5856, 0.5736], [0.5980, 0.5856], [0.5856, 0.5736], [0.5505, 0.5394], [0.4975, 0.4875], [0.4315, 0.4226], [0.3565, 0.3487], [0.2753, 0.2683], [0.1897, 0.1835]],
        [[0.4875, 0.4778], [0.4975, 0.4875], [0.4875, 0.4778], [0.4588, 0.4495], [0.4140, 0.4054], [0.3565, 0.3487], [0.2893, 0.2823], [0.2150, 0.2086], [0.1354, 0.1296]],
        [[0.3887, 0.3805], [0.3970, 0.3887], [0.3887, 0.3805], [0.3644, 0.3565], [0.3258, 0.3184], [0.2753, 0.2683], [0.2150, 0.2086], [0.1472, 0.1413], [0.0734, 0.0680]],
        [[0.2893, 0.2823], [0.2965, 0.2893], [0.2893, 0.2823], [0.2683, 0.2615], [0.2346, 0.2280], [0.1897, 0.1835], [0.1354, 0.1296], [0.0734, 0.0680], [0.0051, 0.0000]]])
    # Actual result
    data = trafo(data)
    # Test
    assert torch.allclose(data["SDF"], expected, atol=1e-4)
    assert data["SDF"][1,1,0] == 1

# def test_normalize_transform():
#     # TODO
#     pass
#     # data = utils.PhysicalVariables(time="now", properties=["test"])
#     # data["test"] = Tensor(np.array([[[[-2, -1, -1],[-1, 0, -1], [-1, -1, -1]]]]))
#     # mean_val = {"test": -1}
#     # std_val = {"test": 0.5}
#     # data_norm = Tensor(np.array([[[[-2, 0, 0],[0, 2, 0], [0, 0, 0]]]]))
#     # transform = trans.NormalizeTransform()
#     # tensor_eq = eq(transform(data, mean_val, std_val)["test"].value, data_norm).flatten
#     # assert tensor_eq

# def test_reduceto2d_transform():
#     # TODO
#     pass
#     # data = utils.PhysicalVariables(time="now", properties=["test"])
#     # data["test"] = zeros([4,2,4])
#     # data_reduced = zeros([1,2,4])
#     # # Actual result
#     # data_actual = trans.ReduceTo2DTransform()(data, loc_hp=[0,1,1])
#     # # Test
#     # assert data_actual["test"].value.shape == data_reduced.shape
#     # tensor_eq = eq(data_actual["test"].value, data_reduced).flatten
#     # assert tensor_eq

# def test_visualize_data():
#     pass

# def test_train_model():
#     pass
#     """ Test input format etc. of train_model"""

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
    test_sdf_transform()