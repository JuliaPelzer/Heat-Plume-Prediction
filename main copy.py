import argparse

import utils.utils_args as ut
import preprocessing.preprocessing as prep
from processing.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, choices=["1hp", "2stages", "allin1", "extend", "test"], default="allin1")
    parser.add_argument("--data_raw", type=str, default="dataset_small_10dp_varyK", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--data_prep", type=str, default=None)
    parser.add_argument("--inputs", type=str, default="gksi") #e.g. "gki", "gksi100", "ogksi1000_finetune", "t", "lmi", "lmik","lmikp", ...
    parser.add_argument("--len_box", type=int, default=64) # for extend:256
    parser.add_argument("--skip_per_dir", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default=None) # required for testing or finetuning
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()


    ut.assertions_args(args)
    ut.make_paths(args) # and check if data / model exists
    ut.save_notes(args)
    ut.save_cla(args)

    # prepare data
    prep.preprocessing(args)

    model = train(args)

    print("Done")