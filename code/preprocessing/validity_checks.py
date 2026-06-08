from torch.utils.data import DataLoader


def receptive_field_is_sufficient(
    receptive_field: int, dataloaders: dict[str, DataLoader], time_step_interval: float = 5.0, resolution: float = 1.0
) -> bool:
    """
    Check if the receptive field is sufficient for the input size of the dataloaders.

    Args:
        max_rf (int): The maximum receptive field of the model.
        dataloaders (dict[str, DataLoader]): A dictionary containing the dataloaders for 'train', 'val', and 'test'.

    Returns:
        bool: True if the receptive field is sufficient, False otherwise.
    """
    max_velocity = get_max_velocity(dataloaders)  # in x-richtung oder integral über breite receptivefield

    required_receptive_field = int((max_velocity * time_step_interval) / resolution)

    return receptive_field >= required_receptive_field


def get_max_velocity(dataloaders: dict[str, DataLoader]) -> float:

    # TODO: Iterate over all dataloader to find the maximum velocity in the input

    return 1.0
