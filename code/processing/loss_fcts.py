from code.utils import logging as log  # noqa: F401

import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class CombiLoss(nn.Module):
    """
    Combines MSE and a secondary loss (e.g. MAE) with ratio alpha.
    Status: Autograd safe. Suitable for training.
    """

    def __init__(self, alpha: float = 1.0, second_loss: nn.Module = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss if second_loss is not None else nn.L1Loss()
        self.alpha = alpha
        self.name = f"CombiLoss (a={alpha}) with {self.secondary_loss_function.__class__.__name__}"

    def forward(self, predictions, labels):
        eval_second = self.secondary_loss_function(predictions, labels)
        return self.alpha * self.mse(predictions, labels) + (1.0 - self.alpha) * eval_second


class SSIMLoss(nn.Module):
    """
    WARNING: returns inverted value compared to old implementation
    Structural Similarity Index Measure.
    Status: Autograd safe. Suitable for training and evaluation.
    """

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, predictions, labels):
        return 1.0 - self.ssim(predictions, labels)


class LinfLoss(nn.Module):
    """
    L-infinity loss (Chebyshev distance).
    Status: Autograd safe (subgradient exists). Suitable for training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.amax(torch.abs(output - target))


class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    pat = torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])
    Status: Evaluation Metric ONLY (Zero gradients). Do NOT use for training.
    """

    def __init__(self, pat_thresholds: list):
        super().__init__()
        self.register_buffer("thresholds", torch.tensor(pat_thresholds).view(1, -1, 1, 1))

    def forward(self, output, label):
        if output.dim() == 3:
            output = output.unsqueeze(1)
            label = label.unsqueeze(1)

        abs_diff = torch.abs(output - label)
        above_thresh = abs_diff > self.thresholds
        pat = above_thresh.to(torch.float32).mean(dim=(2, 3))

        return (pat * 100).mean()
