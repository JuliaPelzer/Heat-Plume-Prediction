import os
import pathlib
from code.utils import logging as log  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import load, save


# Original ConvLSTM cell as proposed by Shi et al. (after Rohit Panda)
def _safe_group_count(num_channels: int, requested_groups: int) -> int:
    groups = min(requested_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return max(groups, 1)


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        frame_size,
        conv_features,
        kernel_sizes,
        norm_type="group",
        groupnorm_groups=8,
    ):

        super().__init__()

        log.info(f"ConvLSTMCell init: in_channels={in_channels}, out_channels={out_channels}, ")
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frame_size = frame_size

        # ---------- Dilated encoder ----------
        self.encoders = nn.ModuleList()

        in_ch = in_channels + out_channels

        # conv_features still controls channel progression
        dilations = [2, 4, 8][: len(conv_features)]

        for i in range(len(conv_features)):
            self.encoders.append(
                Seq2Seq._dilated_block(
                    in_ch,
                    conv_features[i],
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    norm_type=norm_type,
                    groupnorm_groups=groupnorm_groups,
                )
            )
            in_ch = conv_features[i]

        # ---------- Projection to ConvLSTM gates ----------
        self.proj = nn.Conv2d(
            in_channels=in_ch,
            out_channels=4 * out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        # Initialize weights for Hadamard Products (broadcastable across spatial dims)
        self.W_ci = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.W_co = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.W_cf = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # Optional gate/activation logging
        self.log_activations = False
        self.log_every = 100
        self.writer = None
        self.step = 0

    def set_activation_logger(self, writer, log_every: int = 100):
        self.writer = writer
        self.log_every = log_every
        self.log_activations = writer is not None
        log.info("[gate-log] writer set:", self.writer is not None, "log_every", self.log_every)

    def __getstate__(self):
        state = self.__dict__.copy()
        # SummaryWriter holds a thread lock and is not picklable/deepcopy-able
        state["writer"] = None
        return state

    def forward(self, X, H_prev, C_prev):

        x = torch.cat([X, H_prev], dim=1)

        encoded_features = []

        for enc_idx, enc in enumerate(self.encoders):
            x = enc(x)
            encoded_features.append(x)
            if self.log_activations and (self.step % self.log_every == 0) and self.writer is not None:
                with torch.no_grad():
                    self.writer.add_scalar(f"enc/{enc_idx}/mean", x.mean().item(), self.step)
                    self.writer.add_scalar(f"enc/{enc_idx}/std", x.std().item(), self.step)
                    self.writer.add_scalar(f"enc/{enc_idx}/min", x.min().item(), self.step)
                    self.writer.add_scalar(f"enc/{enc_idx}/max", x.max().item(), self.step)
                    self.writer.add_scalar(f"enc/{enc_idx}/var", x.var().item(), self.step)

        conv_output = self.proj(x)
        if self.log_activations and (self.step % self.log_every == 0) and self.writer is not None:
            with torch.no_grad():
                self.writer.add_scalar("proj/mean", conv_output.mean().item(), self.step)
                self.writer.add_scalar("proj/std", conv_output.std().item(), self.step)
                self.writer.add_scalar("proj/min", conv_output.min().item(), self.step)
                self.writer.add_scalar("proj/max", conv_output.max().item(), self.step)
                self.writer.add_scalar("proj/var", conv_output.var().item(), self.step)

        # Split gates
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * torch.tanh(C)

        if self.log_activations and (self.step % self.log_every == 0) and self.writer is not None:
            with torch.no_grad():

                def _log_stats(tag, t):
                    self.writer.add_scalar(f"{tag}/mean", t.mean().item(), self.step)
                    self.writer.add_scalar(f"{tag}/std", t.std().item(), self.step)
                    self.writer.add_scalar(f"{tag}/min", t.min().item(), self.step)
                    self.writer.add_scalar(f"{tag}/max", t.max().item(), self.step)
                    self.writer.add_scalar(f"{tag}/var", t.var().item(), self.step)

                def _log_sat(tag, g):
                    sat = ((g < 0.05) | (g > 0.95)).float().mean().item()
                    self.writer.add_scalar(f"{tag}/sat_frac", sat, self.step)

                _log_stats("gates/i_conv", i_conv)
                _log_stats("gates/f_conv", f_conv)
                _log_stats("gates/o_conv", o_conv)
                _log_stats("gates/C_conv", C_conv)

                _log_stats("gates/input_gate", input_gate)
                _log_stats("gates/forget_gate", forget_gate)
                _log_stats("gates/output_gate", output_gate)

                _log_sat("gates/input_gate", input_gate)
                _log_sat("gates/forget_gate", forget_gate)
                _log_sat("gates/output_gate", output_gate)

                _log_stats("state/C", C)
                _log_stats("state/H", H)

        self.step += 1
        return H, C, encoded_features


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        frame_size,
        prev_boxes,
        extend,
        conv_features,
        kernel_sizes,
        norm_type="group",
        groupnorm_groups=8,
        tbptt_k=2,
    ):

        super().__init__()

        self.out_channels = out_channels
        self.prev_boxes = prev_boxes
        self.extend = extend
        self.tbptt_k = tbptt_k

        # Initialize encoder cell and unroll
        self.convLSTMcell = ConvLSTMCell(
            in_channels,
            out_channels,
            activation,
            frame_size,
            conv_features,
            kernel_sizes,
            norm_type=norm_type,
            groupnorm_groups=groupnorm_groups,
        )

    def forward(self, X, H, C, film_net, scalars):

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        device = X.device
        output = torch.zeros(batch_size, self.out_channels, self.extend, height, width, device=device)

        all_encoded_features = []
        # Unroll over time steps
        for time_step in range(self.prev_boxes, self.prev_boxes + self.extend):
            # Advance through available input timesteps; for forecasting horizons
            # beyond the provided sequence, reuse the last available frame.
            input_idx = min(time_step, seq_len - 1)
            x_t = X[:, :, input_idx, :, :]
            H, C, encoded_feature = self.convLSTMcell(x_t, H, C)

            # FiLM modulation on H
            scalar_t = scalars[:, :, input_idx]  # (B, 3)
            gamma_beta = film_net(scalar_t)  # (B, 2 * film_channels)
            gamma, beta = gamma_beta.chunk(2, dim=1)  # each (B, film_channels)
            H = H * gamma[:, :, None, None] + beta[:, :, None, None]

            all_encoded_features.append(encoded_feature)
            out_idx = time_step - self.prev_boxes
            output[:, :, out_idx] = H

            # Truncate gradients every k steps going *forward*
            if (out_idx + 1) % self.tbptt_k == 0:
                H = H.detach()
                C = C.detach()

        return output, all_encoded_features


class Seq2Seq(nn.Module):
    def __init__(
        self,
        in_channels,
        frame_size,
        prev_boxes,
        extend,
        num_layers,
        enc_conv_features,
        dec_conv_features,
        enc_kernel_sizes,
        dec_kernel_sizes,
        activation="tanh",
        use_groupnorm=True,
        groupnorm_groups=8,
    ):

        super().__init__()

        self.static_channels = 2  # TODO change to 2 (w\o streamlines or 5 w\ streamlines)
        self.dynamic_channels = in_channels - self.static_channels
        self.film_channels = enc_conv_features[-1]

        self.sequential = nn.Sequential()
        self.prev_boxes = prev_boxes
        self.extend = extend
        self.num_layers = num_layers

        self.initial_H_projections = nn.ModuleList(
            [nn.Conv2d(1, enc_conv_features[-1], kernel_size=1) for _ in range(num_layers)]
        )
        self.initial_C_projections = nn.ModuleList(
            [nn.Conv2d(1, enc_conv_features[-1], kernel_size=1) for _ in range(num_layers)]
        )

        if use_groupnorm:
            norm_type = "group"
        else:
            norm_type = "batch"

        static_features = 128
        # Add first ConvLSTM layer
        self.sequential.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=static_features,
                out_channels=enc_conv_features[-1],
                activation=activation,
                frame_size=frame_size,
                prev_boxes=prev_boxes,
                extend=extend,
                conv_features=enc_conv_features,
                kernel_sizes=enc_kernel_sizes,
                norm_type=norm_type,
                groupnorm_groups=groupnorm_groups,
            ),
        )

        if norm_type == "group":
            g = _safe_group_count(enc_conv_features[-1], groupnorm_groups)
            norm_layer = nn.GroupNorm(num_groups=g, num_channels=enc_conv_features[-1])
        else:
            norm_layer = nn.BatchNorm2d(num_features=enc_conv_features[-1])

        self.static_encoder = nn.Sequential(
            Seq2Seq._block(
                self.static_channels,
                static_features,
                kernel_size=3,
                norm_type=norm_type,
                groupnorm_groups=groupnorm_groups,
            ),
            Seq2Seq._block(
                static_features, static_features, kernel_size=3, norm_type=norm_type, groupnorm_groups=groupnorm_groups
            ),
        )
        self.static_features = static_features

        self.film_net = nn.Sequential(
            nn.Linear(self.dynamic_channels, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2 * self.film_channels),
        )

        self.sequential.add_module("batchnorm1", norm_layer)

        # Add rest of the ConvLSTM layers
        for layer in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{layer}",
                ConvLSTM(
                    in_channels=static_features,
                    out_channels=enc_conv_features[-1],
                    activation=activation,
                    frame_size=frame_size,
                    prev_boxes=prev_boxes,
                    extend=extend,
                    conv_features=enc_conv_features,
                    kernel_sizes=enc_kernel_sizes,
                    norm_type=norm_type,
                    groupnorm_groups=groupnorm_groups,
                ),
            )

            if norm_type == "group":
                g = _safe_group_count(enc_conv_features[-1], groupnorm_groups)
                norm_layer = nn.GroupNorm(num_groups=g, num_channels=enc_conv_features[-1])
            else:
                norm_layer = nn.BatchNorm2d(num_features=enc_conv_features[-1])

            self.sequential.add_module(f"batchnorm{layer}", norm_layer)

        # projection between encoder and decoder
        self.enc_dec_projection = nn.Conv2d(
            in_channels=enc_conv_features[-1], out_channels=dec_conv_features[0], kernel_size=1, bias=True
        )

        self.decoder_blocks = nn.ModuleList()

        assert len(dec_conv_features) - 1 <= len(enc_conv_features), (
            f"More decoder blocks ({len(dec_conv_features) - 1}) than encoder levels ({len(enc_conv_features)}) to pair with"
        )

        norm_types = ["group" if norm_type == "group" else "batch" for i in range(len(dec_conv_features) - 2)]
        norm_types.append("none")

        in_ch = dec_conv_features[0] + enc_conv_features[-1]
        for i in range(len(dec_conv_features) - 1):
            in_ch = dec_conv_features[i] + enc_conv_features[-(i + 1)]
            self.decoder_blocks.append(
                Seq2Seq._block(
                    in_channels=in_ch,
                    features=dec_conv_features[i + 1],
                    kernel_size=dec_kernel_sizes[i],
                    norm_type=norm_types[i],
                    groupnorm_groups=groupnorm_groups,
                )
            )

        self.final_conv = nn.Conv2d(dec_conv_features[-1], 1, kernel_size=1, bias=True)

        for proj in self.initial_C_projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

        for proj in self.initial_H_projections:
            nn.init.kaiming_normal_(proj.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
            nn.init.zeros_(proj.bias)

    def forward(self, X, init_frame=None):

        x_in = X
        if init_frame is None:
            # Default: use zeros for initial temperature
            init_frame = torch.zeros(X.shape[0], 1, X.shape[-2], X.shape[-1], device=X.device, dtype=X.dtype)
        else:
            # init_frame should be (B, 1, H, W) - the initial temperature field
            assert init_frame.shape[1] == 1, f"init_frame should have 1 channel, got {init_frame.shape[1]}"

        hidden_H = [self.initial_H_projections[i](init_frame) for i in range(self.num_layers)]
        hidden_C = [self.initial_C_projections[i](init_frame) for i in range(self.num_layers)]

        # --- Static encoder (runs once) ---
        static_input = X[:, :2, 0]  # TODO change to 2 w\o streamlines or 5 w\ streamlines
        static_feat = self.static_encoder(static_input)  # (B, static_features, H, W)

        T = X.shape[2]
        # (B, static_features, T, H, W)
        static_expanded = static_feat.unsqueeze(2).expand(-1, -1, T, -1, -1)

        # Replace X with static features for the recurrent input
        x_in = static_expanded  # ConvLSTM now receives this instead of X

        scalars = X[:, 2:, :, 0, 0]  # TODO change to 2 (w\o streamlines or 5 w\ streamlines)

        X = x_in
        for i in range(self.num_layers):
            X, encoded_features = self.sequential[i * 2](
                X, hidden_H[i], hidden_C[i], self.film_net, scalars
            )  # ConvLSTM layer
            B, C, T, H, W = X.shape
            X = self.sequential[i * 2 + 1](X.view(B * T, C, H, W)).view(B, C, T, H, W)  # Normalization

        output = X

        # get dimensions
        batch_size, _, _, height, width = output.size()

        # initialize decoded output
        decoded_output = torch.zeros(batch_size, 1, self.extend, height, width, device=output.device)

        writer = None
        if (
            hasattr(self.sequential[0], "convLSTMcell")
            and self.sequential[0].convLSTMcell.log_activations
            and (self.sequential[0].convLSTMcell.step % self.sequential[0].convLSTMcell.log_every == 0)
            and self.sequential[0].convLSTMcell.writer is not None
        ):
            writer = self.sequential[0].convLSTMcell.writer
            step = self.sequential[0].convLSTMcell.step

        for pred_box in range(self.extend):
            x = self.enc_dec_projection(output[:, :, pred_box])
            # x = output[:, :, pred_box]
            enc_features_t = encoded_features[pred_box]
            for block_idx, block in enumerate(self.decoder_blocks):
                skip = enc_features_t[-(block_idx + 1)]
                x = torch.cat([x, skip], dim=1)
                x = block(x)
                if writer is not None:
                    with torch.no_grad():
                        writer.add_scalar(f"dec/{pred_box}/{block_idx}/mean", x.mean().item(), step)
                        writer.add_scalar(f"dec/{pred_box}/{block_idx}/std", x.std().item(), step)
                        writer.add_scalar(f"dec/{pred_box}/{block_idx}/min", x.min().item(), step)
                        writer.add_scalar(f"dec/{pred_box}/{block_idx}/max", x.max().item(), step)

            decoded_output[:, :, pred_box] = self.final_conv(x)

        output = decoded_output

        # logging
        if (
            hasattr(self.sequential[0], "convLSTMcell")
            and self.sequential[0].convLSTMcell.log_activations
            and (self.sequential[0].convLSTMcell.step % self.sequential[0].convLSTMcell.log_every == 0)
            and self.sequential[0].convLSTMcell.writer is not None
        ):
            writer = self.sequential[0].convLSTMcell.writer
            step = self.sequential[0].convLSTMcell.step
            with torch.no_grad():

                def _log_stats(tag, t):
                    writer.add_scalar(f"{tag}/mean", t.mean().item(), step)
                    writer.add_scalar(f"{tag}/std", t.std().item(), step)
                    writer.add_scalar(f"{tag}/min", t.min().item(), step)
                    writer.add_scalar(f"{tag}/max", t.max().item(), step)

                _log_stats("io/input", x_in)
                _log_stats("io/output", output)

            writer.add_scalar("final_conv/bias", self.final_conv.bias.item(), step)
            writer.add_scalar("final_conv/weight_mean", self.final_conv.weight.mean().item(), step)
            writer.add_scalar("final_conv/weight_std", self.final_conv.weight.std().item(), step)

        return torch.sigmoid(output)

    def enable_gate_logging(self, writer, log_every: int = 1000):
        for i in range(0, len(self.sequential), 2):
            convlstm = self.sequential[i]
            if hasattr(convlstm, "convLSTMcell"):
                convlstm.convLSTMcell.set_activation_logger(writer, log_every=log_every)

    def save(self, path: pathlib.Path, model_name: str = "model.pt"):
        save(self.state_dict(), path / model_name)

        model_structure = []
        for name, param in self.named_parameters():
            model_structure.append([name, param.shape])
        with open(path / "model_structure.txt", "w") as f:
            f.write(str(model_structure))

    def load(self, model_path: pathlib.Path, device: str, model_name: str = "model.pt"):
        self.load_state_dict(load(model_path / model_name), strict=False)
        self.to(device)

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def infer(self, data, device: str, init_frame=None):
        self.eval()
        torch.set_grad_enabled(False)  # Disable gradient computation for testing
        if init_frame is not None:
            init_frame = init_frame.to(device)
        return torch.sigmoid(self(data.to(device), init_frame)).detach()

    def infer_tiled(
        self,
        data: torch.Tensor,
        device: str,
        init_frame=None,
        tile_size: int = 128,
        overlap: int = 16,
        active_tile_boost: float = 1.0,
        debug_dir: str = "debug_tiles",
    ) -> torch.Tensor:
        self.eval()
        data = data.to(device)

        assert data.shape[0] == 1, f"infer_tiled expects batch size 1, got {data.shape[0]}"

        _, _, _, H, W = data.shape
        grid_step = tile_size // 2 if tile_size <= 2 * overlap else tile_size - 2 * overlap

        output_sum = torch.zeros(1, 1, self.extend, H, W, device=device)
        weight_map = torch.zeros(1, 1, self.extend, H, W, device=device)

        def make_blend_window(size: int, dev) -> torch.Tensor:
            ramp = torch.hann_window(size, periodic=False, device=dev)
            # ramp = ramp.clamp(min=1e-6)
            return torch.outer(ramp, ramp).unsqueeze(0).unsqueeze(0)

        blend = make_blend_window(tile_size, device)

        # --- standard grid tiles ---
        ys = list(range(0, H - tile_size + 1, grid_step))
        xs = list(range(0, W - tile_size + 1, grid_step))
        if not ys or ys[-1] + tile_size < H:
            ys.append(max(0, H - tile_size))
        if not xs or xs[-1] + tile_size < W:
            xs.append(max(0, W - tile_size))

        grid_coords = [(y, x) for y in ys for x in xs]

        # --- extra tiles centered on active pixels in the first channel ---
        active_mask = (data[0, 0] == 1.0).any(dim=0)  # (H, W)
        active_ys, active_xs = torch.where(active_mask)

        extra_coords = set()
        for ay, ax in zip(active_ys.tolist(), active_xs.tolist(), strict=True):
            y0 = int(np.clip(ay - tile_size // 6, 0, H - tile_size))
            x0 = int(np.clip(ax - tile_size // 2, 0, W - tile_size))
            extra_coords.add((y0, x0))

        def save_active_tile(pred: torch.Tensor, y: int, x: int):
            """Save a plot of each time step in the active tile prediction."""
            os.makedirs(debug_dir, exist_ok=True)
            # pred: (1, 1, extend, tile_size, tile_size)
            frames = torch.sigmoid(pred[0, 0]).cpu().numpy()  # (extend, tile_size, tile_size)
            n = len(frames)
            fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
            for t, ax in enumerate(axes[0]):
                im = ax.imshow(frames[t], vmin=0, vmax=1, cmap="hot")
                ax.set_title(f"t={t}")
                ax.axis("off")
            plt.colorbar(im, ax=axes[0][-1], fraction=0.046, pad=0.04)
            fig.suptitle(f"Active tile  y={y}  x={x}", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, f"active_tile_y{y:04d}_x{x:04d}.png"), dpi=100)
            plt.close(fig)

        def run_tile(y, x, boost):
            tile = data[:, :, :, y : y + tile_size, x : x + tile_size]
            init_frame_tile = init_frame[:, :, y : y + tile_size, x : x + tile_size] if init_frame is not None else None
            pred = self(tile, init_frame_tile)
            assert pred.shape[0] == 1, f"infer_tiled expects batch size 1, got {pred.shape[0]}"
            if boost > 1.0:
                save_active_tile(pred, y, x)
            w = blend.expand(1, 1, self.extend, tile_size, tile_size) * boost
            output_sum[:, :, :, y : y + tile_size, x : x + tile_size] += pred * w
            weight_map[:, :, :, y : y + tile_size, x : x + tile_size] += w

        with torch.no_grad():
            for y, x in grid_coords:
                run_tile(y, x, boost=1.0)
            for y, x in extra_coords:
                run_tile(y, x, boost=active_tile_boost)

        output = output_sum / weight_map.clamp(min=1e-08)
        return torch.sigmoid(output).detach()

    @staticmethod
    def _block(in_channels, features, kernel_size=5, norm_type="group", groupnorm_groups=8):
        if norm_type == "group":
            g = _safe_group_count(features, groupnorm_groups)
            norm_layer = nn.GroupNorm(num_groups=g, num_channels=features)
        elif norm_type == "batch":
            norm_layer = nn.BatchNorm2d(num_features=features)
        else:
            norm_layer = None

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, features, kernel_size, padding="same", bias=True)
                self.act1 = nn.LeakyReLU(0.1, inplace=True)
                self.conv2 = nn.Conv2d(features, features, kernel_size, padding="same", bias=True)
                self.norm = norm_layer
                self.act2 = nn.LeakyReLU(0.1, inplace=True)
                self.conv3 = nn.Conv2d(features, features, kernel_size, padding="same", bias=True)
                self.act3 = nn.LeakyReLU(0.1, inplace=True)
                self.skip = (
                    nn.Conv2d(in_channels, features, kernel_size=1, bias=False)
                    if in_channels != features
                    else nn.Identity()
                )

            def forward(self, x):
                residual = self.skip(x)
                x = self.act1(self.conv1(x))
                x = self.conv2(x)
                x = self.act2(self.norm(x) if self.norm is not None else x)
                x = self.conv3(x)
                x = self.act3((self.norm(x) if self.norm is not None else x) + residual)
                return x

        return ResBlock()

    @staticmethod
    def _dilated_block(in_channels, features, kernel_size=3, dilation=1, norm_type="group", groupnorm_groups=8):
        if norm_type == "group":
            g = _safe_group_count(features, groupnorm_groups)
            norm_layer = nn.GroupNorm(num_groups=g, num_channels=features)
        else:
            norm_layer = nn.BatchNorm2d(num_features=features)

        padding = dilation * (kernel_size - 1) // 2

        class DilatedResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    in_channels, features, kernel_size, padding=padding, dilation=dilation, bias=True
                )
                self.act1 = nn.LeakyReLU(0.1, inplace=True)
                self.conv2 = nn.Conv2d(features, features, kernel_size, padding=padding, dilation=dilation, bias=True)
                self.norm = norm_layer
                self.act2 = nn.LeakyReLU(0.1, inplace=True)
                self.conv3 = nn.Conv2d(features, features, kernel_size, padding=padding, dilation=dilation, bias=True)
                self.act3 = nn.LeakyReLU(0.1, inplace=True)
                self.skip = (
                    nn.Conv2d(in_channels, features, kernel_size=1, bias=False)
                    if in_channels != features
                    else nn.Identity()
                )

            def forward(self, x):
                residual = self.skip(x)
                x = self.act1(self.conv1(x))
                x = self.act2(self.norm(self.conv2(x)))
                x = self.act3(self.norm(self.conv3(x)) + residual)
                return x

        return DilatedResBlock()

    def calculate_receptive_field(self) -> dict:
        """
        Analytically computes the receptive field of the full encoder path:

            static_encoder (runs once in Seq2Seq.forward)
                → ConvLSTMCell dilated encoder
                → proj (1x1)

        RF recurrence:
            r[l] = r[l-1] + (k_eff - 1) * jump
            j[l] = j[l-1] * stride
            k_eff = dilation * (kernel - 1) + 1
        """
        cell = self.sequential[0].convLSTMcell

        def _iter_convs(block):
            """Yields all Conv2d layers in forward order."""
            if isinstance(block, nn.Conv2d):
                yield block
            elif hasattr(block, "children"):
                for child in block.children():
                    yield from _iter_convs(child)

        ops = []

        # --- Static encoder (Seq2Seq level, runs once) ---
        for i, block in enumerate(self.static_encoder):
            for layer in _iter_convs(block):
                ops.append((f"static_enc{i}/conv", layer.kernel_size[0], layer.dilation[0], 1))

        # --- Cell dilated encoder (runs every timestep) ---
        for i, enc in enumerate(cell.encoders):
            for layer in _iter_convs(enc):
                ops.append((f"cell_enc{i}/conv", layer.kernel_size[0], layer.dilation[0], 1))

        # --- Projection (1x1, adds nothing to RF) ---
        ops.append(("proj/1x1", 1, 1, 1))

        # --- RF recurrence ---
        r, j = 1, 1
        rows = []

        for label, k, d, s in ops:
            k_eff = d * (k - 1) + 1
            r = r + (k_eff - 1) * j
            j = j * s
            rows.append(
                {
                    "stage": label,
                    "kernel": k,
                    "dilation": d,
                    "stride": s,
                    "k_eff": k_eff,
                    "rf": r,
                    "jump": j,
                }
            )

        # --- Pretty print ---
        # Find where static encoder ends to insert a separator
        static_end = sum(1 for label, _, _, _ in ops if label.startswith("static"))

        log.info(f"\n{'Stage':<25} {'k':>4} {'d':>4} {'s':>4} {'k_eff':>6} {'RF':>6} {'jump':>6}")
        log.info("-" * 60)
        for idx, row in enumerate(rows):
            if idx == static_end:
                log.info("  -- cell encoder (per timestep) --")
            log.info(
                f"{row['stage']:<25} {row['kernel']:>4} {row['dilation']:>4} "
                f"{row['stride']:>4} {row['k_eff']:>6} {row['rf']:>6} {row['jump']:>6}"
            )
        log.info("-" * 60)

        static_rf = rows[static_end - 1]["rf"] if static_end > 0 else 1
        log.info(f"{'Static encoder RF':<25} {static_rf:>6}")
        log.info(f"{'Total RF':<25} {r:>6}  (input size: {cell.frame_size[0]}x{cell.frame_size[1]})")
        log.info(f"RF covers {100 * r / cell.frame_size[0]:.1f}% of the input width\n")

        return {
            "total_rf": r,
            "static_rf": static_rf,
            "stages": rows,
            "input_size": cell.frame_size,
            "rf_coverage_pct": 100 * r / cell.frame_size[0],
        }


def weights_init(m):

    # Convolution layers (Conv2d / ConvTranspose2d)
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # BatchNorm layers
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
        # Start close to identity, small noise helps gradient flow
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

    # ConvLSTM Cell Hadamard weights (W_ci, W_cf, W_co)
    elif isinstance(m, ConvLSTMCell):
        # Keep near zero for stable gate dynamics
        with torch.no_grad():
            oc = m.out_channels
            m.proj.bias[3 * oc :].fill_(1.0)  # output gate
            m.proj.bias[1 * oc : 2 * oc].fill_(1.0)  # forget gate
            # leave input gate and C_conv bias at zero
