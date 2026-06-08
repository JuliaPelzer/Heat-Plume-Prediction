import torch
import torch.nn as nn
import torch.nn.functional as F
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


class WeightedMSE(nn.Module):
    def __init__(self, threshold=0.4, hot_weight=5.0, normalize_by_weights=True):
        super().__init__()
        self.threshold = threshold
        self.hot_weight = hot_weight
        self.normalize_by_weights = normalize_by_weights

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

        w = torch.ones_like(target)
        w[target > self.threshold] = self.hot_weight

        weighted_sq_error = w * (pred - target) ** 2
        if self.normalize_by_weights:
            return weighted_sq_error.sum() / (w.sum() + 1e-12)

        return weighted_sq_error.mean()


class FocalMSE(nn.Module):
    def __init__(
        self,
        gamma=3.0,
    ):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target, logits=False):
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target) ** 2
        weight = (target - 0.5).abs()  # zero weight at background
        weight = weight**self.gamma
        weight = weight / (weight.mean() + 1e-8)  # normalize
        return (weight * error).mean()


class CombinedFocalMSE(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

        weight = (target - 0.5).abs() ** self.gamma  # zero weight at background
        weight = weight / (weight.mean() + 1e-8)  # normalize

        error = (pred - target) ** 2
        focal_mse = (weight * error).mean()

        mse = error.mean()
        return self.alpha * focal_mse + (1 - self.alpha) * mse


class BinaryCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()

        self.gamma = gamma

    def forward(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


class FocalMSE_FP(nn.Module):
    def __init__(self, gamma=2.0, fp_penalty=3.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=False):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target) ** 2

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation**self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return (weight * error).mean()


class FocalMSE_FP_2D(nn.Module):
    def __init__(self, gamma=2.0, fp_penalty=2.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=False):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target) ** 2

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation**self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return weight * error


class FocalMAE_FP(nn.Module):
    def __init__(self, gamma=2.0, fp_penalty=2.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=False):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target).abs()

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation**self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return (weight * error).mean()


class FocalMAE(nn.Module):
    def __init__(self, gamma=2.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.bg_value = bg_value

    def forward(self, pred, target, logits=False):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target).abs()

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        weight = target_deviation**self.gamma

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return (weight * error).mean()


class FocalMAE_FP_2D(nn.Module):
    def __init__(self, gamma=1.0, fp_penalty=3.0, bg_value=0.5):
        """
        gamma      : focuses loss on high-deviation target regions
        fp_penalty : penalty for predicting deviation where target is flat (bg)
        bg_value   : the neutral background value (default 0.5)
        """
        super().__init__()
        self.gamma = gamma
        self.fp_penalty = fp_penalty
        self.bg_value = bg_value

    def forward(self, pred, target, logits=False):
        if logits:
            pred = torch.sigmoid(pred)
        error = (pred - target).abs()

        # How far each pixel is from background — this IS the structure signal
        # 0 at background (0.5), peaks at 1.0 when target is 0 or 1
        target_deviation = (2 * (target - self.bg_value)).abs()  # ∈ [0, 1]
        target_weight = target_deviation**self.gamma

        # False positive: pred deviates from bg, but target does not
        pred_deviation = (2 * (pred - self.bg_value)).abs()
        fp_weight = (pred_deviation * (1 - target_deviation)) ** 2 * self.fp_penalty

        weight = target_weight + fp_weight

        norm = weight.detach().quantile(0.90).clamp(min=1e-6)
        weight = weight / norm

        return weight * error


class MaxNormLoss(nn.Module):
    def forward(self, y_pred, y_true):
        if torch.min(y_pred) < 0:
            y_pred = torch.sigmoid(y_pred)
        return (y_pred - y_true).abs().max()


class MAE_Logits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        error = (pred - target).abs()
        return error.mean()
