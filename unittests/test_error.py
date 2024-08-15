import torch
import math
from postprocessing.measurements import percentage_misclassification

def test_error():
    tol = 1e-9

    # Fixture
    tensor = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]]])
    # Tensor 2 compare
    pred_tensor = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]]])
    # Test
    assert abs(percentage_misclassification(tensor, pred_tensor, 0.1) - 0.) <  tol

    # Fixture
    tensor = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]]])
    # Tensor 2 compare
    pred_tensor = torch.tensor([[[[0,0.09,0],[0,0.07,0],[0,0,0]],[[1,0.91,1],[1,1,1],[1,0.95,1]]]])
    # Test
    assert abs(percentage_misclassification(tensor, pred_tensor, 0.1) - 0.) < tol

    # Fixture
    tensor = torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]]]])
    # Tensor 2 compare
    pred_tensor = torch.tensor([[[[0,0,0],[0,10,0],[0,-5,0]],[[1,4.6,1],[1,1,1],[1,23,1]]]])
    # Test
    assert abs(percentage_misclassification(tensor, pred_tensor, 0.1) - 4/math.prod(tensor.shape)) < tol

if __name__ == "__main__":
    test_error()