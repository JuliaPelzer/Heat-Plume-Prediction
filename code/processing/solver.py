import gc
import logging
import time
from code.postprocessing.visualization import visualize_outputs
from code.preprocessing.datasets.dataset import DatasetType
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
from torch import manual_seed
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

    def __post_init__(self):
        if not self.finetune:
            if isinstance(self.model, Seq2Seq):
                self.model.apply(convlstm_weights_init)
            else:
                self.model.apply(model_weights_init)
        self.metrics: dict = {
            "Huber": HuberLoss(),
        }

    def train(self, datasetType: DatasetType, train_dataloader: DataLoader, val_dataloader: DataLoader, args: dict):
        manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        writer = SummaryWriter(args["destination"])
        device = args["device"]

        # Calculate parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        log.info(f"Total Parameters: {total_params:,}")
        log.info(f"Trainable Parameters: {trainable_params:,}")

        log.info(f"Total Parameters: {total_params:,}, Trainable: {trainable_params:,}")

        # if optimizer_switch is True, switch to LBFGS optimizer after 90% of epochs
        self.epoch_switch_optimizer = args["epochs"] + 1
        if self.optimizer_switch:
            self.epoch_switch_optimizer = int(0.9 * self.epoch_switch_optimizer)

        epochs = tqdm(range(args["epochs"]), "CNN Training Epochs", dynamic_ncols=True, unit="epoch", leave=True)
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
                    self.opt = LBFGS(self.model.parameters(), history_size=20, line_search_fn="strong_wolfe")
                    log.info(f"Switched to LBFGS optimizer at epoch {epoch}.")

                # Training
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(train_dataloader, device)

                # Validation
                self.model.eval()
                if False:
                    for m in self.model.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            m.train()  # Force BN to use the current batch's stats
                val_epoch_loss, other_losses_val = self.run_epoch(val_dataloader, device)

                if False:  # realK
                    val_epoch_loss = other_losses_val["Huber"]  # TODO for realK

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
                    f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {current_lr:.1e}"
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
                if self.best_model_params is not None:
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

    def run_epoch(self, dataloader: DataLoader, device: str):
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in self.metrics}

        gc.collect()
        torch.cuda.empty_cache()

        # Helper to eliminate 4x duplicated forward/cropping logic
        def _forward_pass(x_batch, y_batch):
            pred = self.model(x_batch)
            req_h, req_w = pred.shape[2:]
            sh, sw = (y_batch.shape[2] - req_h) // 2, (y_batch.shape[3] - req_w) // 2
            reduced = y_batch[:, :, sh : sh + req_h, sw : sw + req_w]
            return pred, reduced, self.loss_func(pred, reduced)

        for _batch_idx, (x, y) in tqdm(
            enumerate(dataloader),
            "Processing batches",
            total=len(dataloader),
            dynamic_ncols=True,
            unit="batch",
            leave=False,
            mininterval=10,
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.set_grad_enabled(self.model.training):
                if self.model.training:
                    self.opt.zero_grad(set_to_none=True)

                    if False:
                      pass
                    # if self.opt.__class__.__name__ == "LBFGS":
                    #     def closure():
                    #         self.opt.zero_grad()
                    #         _, _, loss = _forward_pass(x, y)
                    #         loss.backward()
                    #         return loss

                    #     self.opt.step(closure)

                    #     with torch.no_grad():
                    #         y_pred, y_reduced, loss = _forward_pass(x, y)
                    else:
                        y_pred, y_reduced, loss = _forward_pass(x, y)
                        loss.backward()
                        self.opt.step()
                else:
                    y_pred, y_reduced, loss = _forward_pass(x, y)

            epoch_loss += loss.item()

            with torch.no_grad():
                for name, metric in self.metrics.items():
                    epoch_metrics[name] += metric(y_pred, y_reduced).item()

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
