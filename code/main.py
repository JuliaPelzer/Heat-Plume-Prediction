import optuna
from optuna.trial import TrialState
import argparse
from pathlib import Path
from processing.training import training, run
from utils.utils_args import read_cla

PATH_MODELS_DIR = Path("../runs/bm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--hsearch", action="store_true", help="perform hyperparameter search", default=False)
    args = parser.parse_args()
    args = vars(args)
    hsearch = args["hsearch"]

    args = read_cla(PATH_MODELS_DIR / args["destination"])
    
    if not hsearch:
        model = training(args)

    else:
        print("Study name: ", args["destination"])
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{PATH_MODELS_DIR}/STUDY.db", study_name="STUDY_NAME", load_if_exists=True)
        study.optimize(lambda trial: run(trial, args), n_trials=30)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    print("Done")