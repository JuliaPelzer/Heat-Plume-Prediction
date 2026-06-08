from code.preprocessing.datasets.dataset import DataPoint, DatasetBasis
from code.preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts, SimulationDatasetCutsSequential
from code.utils import logging as log  # noqa: F401

from torch.utils.data import DataLoader


def init_data(
    args: dict, datapoint_test: list, datapoint_validate: list, datapoint_train: list, tmp_bool_cutouts: bool = False
):
    datasets = {}
    network = args.get("network", "unet").lower()

    if not tmp_bool_cutouts or args["case"] == "test":  # NO CUTOUTS
        dataset_train = DataPoint(args["data_prep"], i=datapoint_train)
    else:  # DO CUTOUTS
        if network in ["convlstm", "rnn", "lstm"]:
            dataset_train = SimulationDatasetCutsSequential(
                args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=datapoint_train
            )
        else:
            dataset_train = SimulationDatasetCuts(
                args["data_prep"], skip_per_dir=args["skip_per_dir"], ids=datapoint_train, box_size=args["len_box"]
            )

    if network in ["convlstm", "rnn", "lstm"]:
        dataset_val = SimulationDatasetCutsSequential(
            args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=datapoint_validate[0]
        )
    else:
        dataset_val = DataPoint(args["data_prep"], i=datapoint_validate)
    dataset_test = DataPoint(args["data_prep"], i=datapoint_test)

    datasets["train"] = dataset_train
    datasets["val"] = dataset_val
    datasets["test"] = dataset_test

    return datasets["train"].input_channels, datasets["train"].output_channels, datasets


def construct_dataloader(batch_size: int, dataset: DatasetBasis, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=min(len(dataset), batch_size),
        shuffle=shuffle,
        drop_last=True,
        num_workers=4,  # new, before: 0
        pin_memory=True,  # new
        persistent_workers=True,  # new: makes progress approx. 10 times faster
        prefetch_factor=4,  # new
    )
