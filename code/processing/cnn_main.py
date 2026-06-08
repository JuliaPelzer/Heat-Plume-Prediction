from code.processing.training import training
from code.utils import logging as log  # noqa: F401
from code.utils.yaml_parser import MLStepConfig, Paths, RunConfiguration

import optuna
from optuna.trial import TrialState


def step_cnn(run_configuration: RunConfiguration, paths: Paths, step_config: MLStepConfig, step_name: str, mode: str):
    args = {}
    args["case"] = mode
    args["device"] = run_configuration.device
    args["model"] = paths.results / run_configuration.run_name / step_name
    args["data_prep"] = paths.datasets_prep
    args["data_raw"] = paths.datasets_raw / run_configuration.dataset
    args["destination"] = paths.results / run_configuration.run_name / step_name

    args["visualize"] = step_config.general.visualize
    args["previous_results"] = step_config.previous_results
    args["epochs"] = step_config.general.epochs
    args["datapoint_test"] = step_config.datapoints.test
    args["datapoint_validate"] = step_config.datapoints.validation
    args["datapoint_train"] = step_config.datapoints.train
    args["scheduler"] = step_config.scheduler

    args["destination"].mkdir(parents=True, exist_ok=True)

    if mode == "hopt":
        log.info("Study name: ", args["destination"])
        study = optuna.create_study(
            direction="minimize",
            storage=f"sqlite:///{args['destination']}/TEST_STUDY.db",
            study_name="NAME",
            load_if_exists=True,
        )

        def run_hopt(trial):
            args_copy = args.copy()
            args_copy["destination"] = args["destination"] / f"trial_{trial.number}"
            args_copy["destination"].mkdir(parents=True, exist_ok=True)
            args_copy["visualize"] = False
            args_copy["case"] = "train"

            hopt = step_config.hopt_parameters
            args_copy["len_box"] = trial.suggest_categorical("len_box", hopt.len_box)
            args_copy["skip_per_dir"] = trial.suggest_categorical("skip_per_dir", hopt.skip_per_dir)
            args_copy["stride"] = trial.suggest_categorical("stride", hopt.stride)
            args_copy["dilation"] = trial.suggest_categorical("dilation", hopt.dilation)
            args_copy["activation_fct"] = trial.suggest_categorical("activation_fct", hopt.activation)
            args_copy["norm"] = trial.suggest_categorical("norm", hopt.norm)
            args_copy["repeat_inner"] = trial.suggest_categorical("repeat_inner", hopt.repeat_inner)
            args_copy["optimizer_switch"] = trial.suggest_categorical("optimizer_switch", hopt.optimizer_switch)
            args_copy["bool_cutouts"] = trial.suggest_categorical("bool_cutouts", hopt.bool_cutouts)
            args_copy["batchsize"] = trial.suggest_categorical("batchsize", hopt.batchsize)
            args_copy["depth"] = trial.suggest_categorical("depth", hopt.depth)
            args_copy["init_features"] = trial.suggest_categorical("init_features", hopt.init_features)
            args_copy["kernel_size"] = trial.suggest_categorical("kernel_size", hopt.kernel_size)
            args_copy["inputs"] = trial.suggest_categorical("inputs", ["".join(item) for item in hopt.inputs])
            args_copy["train_loss"] = trial.suggest_categorical("train_loss", hopt.train_loss)

            return training(args_copy)

        study.optimize(run_hopt, n_trials=100)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        log.info("Study statistics: ")
        log.info("  Pruned trials: ", len(pruned_trials))
        log.info("  Complete trials: ", len(complete_trials))
        log.info("  Number of finished trials: ", len(study.trials))

        log.info("Best trial:")
        trial = study.best_trial
        log.info("  Value: ", trial.value)

        log.info("  Params: ")
        for key, value in trial.params.items():
            log.info(f"    {key}: {value}")
    else:
        parameter = step_config.model_parameters

        args["network"] = parameter.network.lower()
        args["inputs"] = "".join(parameter.inputs)
        args["outputs"] = "".join(parameter.outputs)
        args["batchsize"] = parameter.batchsize
        args["stride"] = parameter.stride
        args["skip_per_dir"] = parameter.skip_per_dir
        args["len_box"] = parameter.len_box
        args["train_loss"] = parameter.train_loss
        args["bool_cutouts"] = parameter.bool_cutouts
        args["optimizer_switch"] = parameter.optimizer_switch
        args["optimizer"] = parameter.optimizer
        args["activation_fct"] = parameter.activation

        if parameter.network == "unet":
            args["kernel_size"] = parameter.kernel_size
            args["depth"] = parameter.depth
            args["init_features"] = parameter.init_features
            args["dilation"] = parameter.dilation
            args["norm"] = parameter.norm
            args["repeat_inner"] = parameter.repeat_inner
        elif parameter.network in ["convlstm", "rnn", "lstm"]:
            args["visualize_interval"] = parameter.visualize_interval
            args["overfit"] = parameter.overfit
            args["overfit_on"] = parameter.overfit_on
            args["time_steps_to_predict"] = parameter.time_steps_to_predict
            args["num_layers"] = parameter.rnn_num_layers
            args["enc_conv_features"] = parameter.enc_conv_features
            args["dec_conv_features"] = parameter.dec_conv_features
            args["enc_kernel_sizes"] = parameter.enc_kernel_sizes
            args["dec_kernel_sizes"] = parameter.dec_kernel_sizes
            args["max_simulation_timestep"] = parameter.max_simulation_timestep

        training(args)
