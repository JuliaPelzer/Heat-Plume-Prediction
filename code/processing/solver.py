import gc
import logging
import time
from code.postprocessing.visualization import visualize_inputs, visualize_outputs
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.datasets.dataset_cuts_jit import DefaultBatchContext, SublistBatchContext
from code.processing.loss_fcts import LinfLoss, PATLoss, SSIMLoss
from code.processing.networks.convLSTM import Seq2Seq
from code.processing.networks.convLSTM import weights_init as convlstm_weights_init
from code.processing.networks.model import weights_init as model_weights_init
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import save_yaml
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import manual_seed, nn
from torch.nn import HuberLoss, L1Loss, Module, MSELoss, modules
from torch.optim import LBFGS, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class Solver:
    model: Module
    train_dataset: Dataset
    val_dataset: Dataset
    loss_func: modules.loss._Loss = MSELoss()
    batchsize: int = 32
    opt: Optimizer = AdamW
    optimizer_switch: bool = False
    finetune: bool = False
    best_model_params: dict = None
    metrics: dict = None
    global_step: int = 0

    def __post_init__(self):
        self.sequence_context = SublistBatchContext() if isinstance(self.model, Seq2Seq) else DefaultBatchContext()

        if not self.finetune:
            if isinstance(self.model, Seq2Seq):
                self.model.apply(convlstm_weights_init)
                nn.init.xavier_uniform_(self.model.final_conv.weight)
                nn.init.constant_(self.model.final_conv.bias, 0.0)
            else:
                self.model.apply(model_weights_init)

        self.metrics: dict = {
            "Huber": HuberLoss(),
        }

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_to_pred(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Centre-crop y so its spatial dims match y_pred."""
        required_size = y_pred.shape[2:]
        start_h = (y.shape[2] - required_size[0]) // 2
        start_w = (y.shape[3] - required_size[1]) // 2
        return y[
            :,
            :,
            start_h : start_h + required_size[0],
            start_w : start_w + required_size[1],
        ]

    def _run_split_eval(self, dataloader: DataLoader, device: str):
        """Run run_epoch in eval mode with no_grad, regardless of current state."""
        self.model.eval()
        with torch.no_grad():
            return self.run_epoch(dataloader, device, writer=None)

    def _step_scheduler_and_log(
        self,
        scheduler,
        scheduler_type: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ):
        old_lr = self.opt.param_groups[0]["lr"]
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(train_loss)
            new_lr = self.opt.param_groups[0]["lr"]
            logging.info(
                f"[LR CHECK] epoch={epoch} train_loss={train_loss:.4e} val_loss={val_loss:.4e} "
                f"best={scheduler.best:.4e} bad_epochs={scheduler.num_bad_epochs}/{scheduler.patience} "
                f"lr={old_lr:.2e}→{new_lr:.2e}"
            )
        else:
            scheduler.step()
            new_lr = self.opt.param_groups[0]["lr"]
            logging.info(
                f"[LR CHECK] epoch={epoch} train_loss={train_loss:.4e} val_loss={val_loss:.4e} lr={new_lr:.2e}"
            )

    # ------------------------------------------------------------------

    def save_epoch_metrics_yaml(
        self,
        destination,
        filename,
        epoch,
        train_epoch_loss,
        val_epoch_loss,
        other_losses_train,
        other_losses_val,
        no_params=None,
        max_epochs=None,
        training_time=None,
        checkpoint_type=None,
    ):
        metrics = {
            "current_epoch": epoch,
            "train": {**dict(other_losses_train), "train loss": train_epoch_loss},
            "val": {**dict(other_losses_val), "val loss": val_epoch_loss},
        }
        for key, value in [
            ("checkpoint_type", checkpoint_type),
            ("no_params", no_params),
            ("max_epochs", max_epochs),
            ("training_time [s]", training_time),
        ]:
            if value is not None:
                metrics[key] = value

        save_yaml(metrics, destination / filename)

    def log_receptive_field_info(self):
        try:
            rf_info = self.model.calculate_receptive_field()
            log.info(f"[RECEPTIVE FIELD] Stage{'':<21} k    d    s   k_eff    RF   jump")
            log.info(f"[RECEPTIVE FIELD] {'-' * 58}")
            for stage in rf_info["stages"]:
                log.info(
                    f"[RECEPTIVE FIELD] {stage['stage']:<25} {stage['kernel']:>4} {stage['dilation']:>4} "
                    f"{stage['stride']:>4} {stage['k_eff']:>6} {stage['rf']:>6} {stage['jump']:>6}"
                )
            log.info(f"[RECEPTIVE FIELD] {'-' * 58}")
            log.info(
                f"[RECEPTIVE FIELD] Total RF{'':<17} {rf_info['total_rf']:>6}  "
                f"(input size: {rf_info['input_size'][0]}x{rf_info['input_size'][1]})"
            )
            log.info(f"[RECEPTIVE FIELD] RF covers {rf_info['rf_coverage_pct']:.1f}% of the input width")
        except Exception as e:
            log.warning(f"Could not calculate receptive field: {e}")

    def train(
        self,
        datasetType: DatasetType,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        args: dict,
    ):
        manual_seed(0)
        start_time = time.perf_counter()

        overfit_on: int | None = None
        if "overfit" in args:
            overfit_on = args["overfit_on"]

        vis_interval: int | None = None
        if "visualize_interval" in args:
            vis_interval = args["visualize_interval"]

        # initialize tensorboard
        writer = SummaryWriter(args["destination"])
        device = args["device"]

        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        log.info(f"Total Parameters: {total_params:,}, Trainable: {trainable_params:,}")

        if hasattr(self.model, "calculate_receptive_field"):
            self.log_receptive_field_info()

        # if optimizer_switch is True, switch to LBFGS optimizer after 90% of epochs
        self.epoch_switch_optimizer = args["epochs"] + 1
        if self.optimizer_switch:
            self.epoch_switch_optimizer = int(0.9 * self.epoch_switch_optimizer)

        epochs = tqdm(
            range(args["epochs"]),
            "CNN Training Epochs",
            dynamic_ncols=True,
            unit="epoch",
            leave=True,
        )
        self.best_model_params = None

        # Assume noisy data
        scheduler = args["scheduler"]
        self.opt = self.opt(self.model.parameters(), scheduler.init_lr, weight_decay=1e-4)

        if scheduler.type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                self.opt,
                mode=scheduler.mode,
                factor=scheduler.factor,
                patience=scheduler.patience,
                threshold=scheduler.threshold,
                min_lr=scheduler.min_lr,
            )
        elif scheduler.type == "StepLR":
            scheduler = StepLR(self.opt, step_size=scheduler.step_size, gamma=scheduler.gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {args['scheduler'].type}")

        early_stop_patience = scheduler.patience * 2
        early_stop_counter = 0
        min_delta = 1e-6

        try:
            for epoch in epochs:
                if epoch == self.epoch_switch_optimizer:
                    self.opt = LBFGS(
                        self.model.parameters(),
                        history_size=20,
                        line_search_fn="strong_wolfe",
                    )
                    log.info(f"Switched to LBFGS optimizer at epoch {epoch}.")

                # Training
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(train_dataloader, device, overfit_on)

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_epoch_loss, other_losses_val = self.run_epoch(val_dataloader, device, overfit_on)

                scheduler.step(val_epoch_loss)

                # Logging
                for metric_name, metric_value in other_losses_val.items():
                    writer.add_scalar(f"val {metric_name}", metric_value, epoch)
                for metric_name, metric_value in other_losses_train.items():
                    writer.add_scalar(f"train {metric_name}", metric_value, epoch)
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)

                current_lr = self.opt.param_groups[0]["lr"]
                epochs.set_postfix_str(
                    f"train loss: {train_epoch_loss:.4e}, val loss: {val_epoch_loss:.4e}, lr: {current_lr:.2e}"
                )

                # Keep best model
                if self.best_model_params is None or val_epoch_loss < (self.best_model_params["loss"] - min_delta):
                    self.best_model_params = {
                        "epoch": epoch,
                        "loss": val_epoch_loss,
                        "train loss": train_epoch_loss,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        # "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    log.info(f"No improvement for {early_stop_counter}/{early_stop_patience} epochs.")

                    if early_stop_patience <= early_stop_counter:
                        log.info(
                            f"\nEarly stopping triggered! No improvement in validation loss for {early_stop_patience} consecutive epochs."
                        )
                        break
                if self.best_model_params is not None and vis_interval is not None and epoch % vis_interval == 0:
                    with torch.no_grad():
                        model_tmp = deepcopy(self.model)
                        visualize_outputs(
                            datasetType,
                            model_tmp,
                            val_dataloader,
                            args,
                            plot_path=args["destination"] / f"temp{epoch}",
                            amount_datapoints_to_visu=1,
                            pic_format="png",
                        )

                # --- Visualization ---
                if vis_interval is not None and epoch % vis_interval == 0:
                    with torch.no_grad():
                        best_model_tmp = deepcopy(self.model)
                        best_model_tmp.load_state_dict(self.best_model_params["state_dict"])
                        best_model_tmp.to(args["device"])
                        plot_base = args["destination"] / f"train_best_e{epoch}"
                        plot_base = args["destination"] / f"train_best_e{epoch}"
                        if epoch == 0:
                            visualize_inputs(
                                train_dataloader,
                                args,
                                amount_datapoints_to_visu=1,
                                plot_path=args["destination"] / f"train_temp{epoch}",
                                pic_format="png",
                            )
                            visualize_outputs(
                                best_model_tmp,
                                train_dataloader,
                                args,
                                plot_path=plot_base,
                                amount_datapoints_to_visu=1,
                                pic_format="png",
                                plot_true=True,
                            )
                        else:
                            visualize_outputs(
                                best_model_tmp,
                                train_dataloader,
                                args,
                                plot_path=plot_base,
                                amount_datapoints_to_visu=1,
                                pic_format="png",
                                plot_true=False,
                            )

        except KeyboardInterrupt:
            log.info("\nTraining interrupted by user.")
            try:
                with torch.no_grad():
                    model_tmp = deepcopy(self.model)
                    model_tmp.load_state_dict(self.best_model_params["state_dict"])
                    model_tmp.to(args["device"])
                    model_tmp.save(args["destination"], model_name=f"interim_model_e{epoch}.pt")
                    visualize_outputs(
                        datasetType,
                        model_tmp,
                        val_dataloader,
                        args,
                        plot_path=args["destination"] / f"plot_val_interim_e{epoch}",
                        amount_datapoints_to_visu=2,
                        pic_format="png",
                    )
            except Exception as e:
                logging.error(e)
            try:
                choice = input("Enter new LR to continue, or press Enter to stop: ")
                if choice:
                    new_lr = float(choice)
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    log.info(f"Resuming with new LR: {new_lr}")
            except Exception:
                pass
        finally:
            writer.close()

        if self.best_model_params is not None:
            self.model.load_state_dict(self.best_model_params["state_dict"])
            self.opt.load_state_dict(self.best_model_params["optimizer"])
            log.info(f"Best model was found in epoch {self.best_model_params['epoch']}.")
            return self.best_model_params["loss"]
        else:
            logging.warning("Training stopped before any model could be saved.")
            return float("inf")

    def run_epoch(self, dataloader: DataLoader, device: str, overfit_on: int | None):
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in self.metrics}

        gc.collect()
        torch.cuda.empty_cache()

        if overfit_on is not None:
            y_pred = y_reduced = None
            for i, (x, y) in enumerate(dataloader):
                if i not in overfit_on:
                    continue
                x, y = x.to(device), y.to(device)
                self.opt.zero_grad()
                y_pred = self.model(x)
                y_reduced = self._crop_to_pred(y, y_pred)
                loss = self.loss_func(y_pred, y_reduced)
                if self.model.training:
                    loss.backward()
                    self.opt.step()
                    self.global_step += 1
                epoch_loss += loss.detach().item()
                if i not in overfit_on:
                    continue
                x, y = x.to(device), y.to(device)
                self.opt.zero_grad()
                y_pred = self.model(x)
                y_reduced = self._crop_to_pred(y, y_pred)
                loss = self.loss_func(y_pred, y_reduced)
                if self.model.training:
                    loss.backward()
                    self.opt.step()
                    self.global_step += 1
                epoch_loss += loss.detach().item()

            epoch_loss /= len(overfit_on)
            metric_values = {}
            if y_pred is not None and y_reduced is not None:
                metric_values = {
                    name: metric(y_pred, y_reduced).detach().item() for name, metric in self.metrics.items()
                }
            return epoch_loss, metric_values

        # --- Normal (non-overfit) path ---
        for _batch_idx, batch in tqdm(
            enumerate(dataloader),
            "Processing batches",
            total=len(dataloader),
            dynamic_ncols=True,
            unit="batch",
            leave=False,
            mininterval=10,
        ):
            if len(batch) == 3:
                x, y, metadata_list = batch
            else:
                x, y = batch
                metadata_list = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            init_frame = None

            if metadata_list is not None:
                init_frame = self.sequence_context.build_init_frame(x, metadata_list)

            with torch.set_grad_enabled(self.model.training):
                if self.model.training:
                    self.opt.zero_grad(set_to_none=True)

                y_pred = self.model(x) if init_frame is None else self.model(x, init_frame)
                req_h, req_w = y_pred.shape[-2:]
                sh, sw = (y.shape[-2] - req_h) // 2, (y.shape[-1] - req_w) // 2
                y_reduced = y[..., sh : sh + req_h, sw : sw + req_w]
                loss = self.loss_func(y_pred, y_reduced)

                if self.model.training:
                    loss.backward()
                    self.opt.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                for name, metric in self.metrics.items():
                    epoch_metrics[name] += metric(y_pred, y_reduced).item()

            if metadata_list is not None:
                self.sequence_context.update(metadata_list, y_pred)

        num_batches = len(dataloader)
        return epoch_loss / num_batches, {k: v / num_batches for k, v in epoch_metrics.items()}

    def save_metrics_separate_yaml(self, dataloaders: dict, destination: Path, device: str, all_metrics: dict):
        metrics = {}
        self.model.eval()
        loss_funcs = {
            "Huber": HuberLoss().to(device),
            "Linf": LinfLoss().to(device),
            "MAE": L1Loss().to(device),
            "MSE": MSELoss().to(device),
            "SSIM": SSIMLoss().to(device),
        }

        with torch.no_grad():
            for case, dataloader in dataloaders.items():
                norm = dataloader.dataset.norm
                metrics[case] = {m: [] for m in ["Huber", "Linf", "MAE", "MSE", "PAT", "SSIM"]}
                pat_loss = None

                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    y_pred = self.model(x)

                    num_channels = y_pred.shape[1]

                    if pat_loss is None:
                        pat_loss = PATLoss(pat_thresholds=[0.1] * num_channels).to(device)
                        loss_funcs["PAT"] = pat_loss

                    req_h, req_w = y_pred.shape[2:]
                    h, w = y.shape[2:]
                    sh, sw = (h - req_h) // 2, (w - req_w) // 2
                    y_reduced = y[:, :, sh : sh + req_h, sw : sw + req_w]

                    metrics[case]["SSIM"].append(1.0 - loss_funcs["SSIM"](y_pred, y_reduced).item())

                    for b in range(y_pred.shape[0]):
                        norm.reverse(y_pred[b], data_type="Labels")
                        norm.reverse(y_reduced[b], data_type="Labels")

                    for m_name in ["Huber", "Linf", "MAE", "MSE"]:
                        c_vals = [
                            loss_funcs[m_name](y_pred[:, c : c + 1], y_reduced[:, c : c + 1]).item()
                            for c in range(num_channels)
                        ]
                        metrics[case][m_name].append(c_vals)

                    metrics[case]["PAT"].append(loss_funcs["PAT"](y_pred, y_reduced).item())

                for m_name in loss_funcs.keys():
                    if metrics[case][m_name]:
                        val_tensor = torch.tensor(metrics[case][m_name], dtype=torch.float32)
                        metrics[case][m_name] = val_tensor.mean(dim=0).tolist()

        all_metrics["loss"] = metrics
        save_yaml(all_metrics, destination / "measurements.yaml")


def log_grad_stats(model, logger=logging):
    logger.info("Gradient and Parameter statistics:")
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        logger.info(
            f"[GRAD]  {name:40s} min={grad.min():+.2e} max={grad.max():+.2e} "
            f"mean={grad.mean():+.2e} std={grad.std():+.2e}"
        )
        logger.info(f"[PARAM] {name:40s} min={p.min():+.2e} max={p.max():+.2e} mean={p.mean():+.2e} std={p.std():+.2e}")
