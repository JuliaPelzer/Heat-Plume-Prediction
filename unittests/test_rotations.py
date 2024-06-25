import torch
import os
import processing.rotation as rt

def test_rotation():
    # Get path to info file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    info_path = os.path.join(current_dir, 'dummy_files', 'info.yaml')

    # Input rotation
    # Fixture
    data = torch.Tensor([[[0,0],[0,0]],
                         [[1,1],[1,1]],
                         [[1,2],[3,4]]])
    
    # Expected result
    expected = torch.Tensor([[[0,0],[0,0]],
                             [[-1,-1], [-1,-1]],
                             [[4,3],[2,1]]])
    # Actual result
    data = rt.rotate(data,180,info_path)

    # Test
    assert torch.allclose(data, expected), "Problem detected in input rotation!"


    # Label rotation
    # Fixture
    data = torch.Tensor([[[1,2],[3,4]]])
    
    # Expected result
    expected = torch.Tensor([[[4,3],[2,1]]])
    # Actual result
    data = rt.rotate(data,180,info_path)
    
    # Test
    assert torch.allclose(data, expected), "Problem detected in label rotation!"

def test_get_rot_angle():
    # rotation < 180
    # Fixture
    a = [1,0]
    b = [0,1]

    # Expected result
    exp_angle = 90
    # Actual result
    calc_angle = rt.get_rotation_angle(a, b)

    # Test
    assert calc_angle == exp_angle, "Problem detected in finding angle < 180"


    # rotation > 180
    # Fixture
    a = [0,1]
    b = [1,0]

    # Expected result
    exp_angle = 270
    # Actual result
    calc_angle = rt.get_rotation_angle(a, b)

    # Test
    assert calc_angle == exp_angle, "Problem detected in finding angle < 180"


if __name__ == "__main__":
    test_rotation()
    test_get_rot_angle()