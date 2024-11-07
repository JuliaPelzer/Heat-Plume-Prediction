import torch
from torch import nn


class CausalConv2d(nn.Module):
    def __init__(self, level: int = 0, kernel_size_x: int = 3, in_channels: int = 1, out_channels: int = 32) -> None:
        dilation = 2**level
        super().__init__()
        self.padding_x = (kernel_size_x - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size_x, 3),
            padding=(self.padding_x, 1),
            dilation=(dilation, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        return x[:, :, : -self.padding_x, :]


class Encoder(nn.Module):
    def __init__(self, levels: int = 1, kernel_size_x: int = 3) -> None:
        super().__init__()
        self.convs = nn.Sequential(CausalConv2d(level=0, kernel_size_x=kernel_size_x, in_channels=1, out_channels=32),
                                   *[CausalConv2d(level=i, kernel_size_x=kernel_size_x, in_channels=32*2**(i-1), out_channels=32*2**i) for i in range(1, levels)])

    def forward(self, x):
        return self.convs(x)


# %%
class AutoRegressive(nn.Module):
    def __init__(self, levels: int = 2, kernel_size_x: int = 3) -> None:
        super().__init__()
        self.temp_enc = Encoder(levels=levels, kernel_size_x=kernel_size_x)
        self.perm_enc = Encoder(levels=levels, kernel_size_x=kernel_size_x)
        self.decoder = nn.Sequential(CausalConv2d(level=levels, kernel_size_x=kernel_size_x, in_channels=32*2**levels, out_channels=1))

    def forward(self, inputs: torch.Tensor, label: torch.Tensor):
        """

        inputs (=permeability) shape: (batch, 1, length, width)
        label (=temperature) shape: (batch, 1, length, width)
        """
        temperature = label #input["temperature"]
        permeability = inputs #input["permeability"]
        assert  permeability.shape[1] == 1, "Input (=Permeability) should have 1 channel"

        temp_enc = self.temp_enc(temperature[:, :, :-1, :])
        perm_enc = torch.flip(self.perm_enc(torch.flip(permeability, (2,))), (2,))

        temperature = temperature[:, :, 1:, :]
        perm_enc = perm_enc[:, :, 1:, :]

        prediction = nn.functional.tanh(temp_enc + perm_enc).add(1).div(2)

        return prediction

    def get_loss(self, inputs: torch.Tensor, label: torch.Tensor, loss_fn: nn.Module):
        """

        inputs (=permeability) shape: (batch, 1, length, width)
        label (=temperature) shape: (batch, 1, length, width)
        """
        assert  inputs.shape[1] == 1, "Input (=Permeability) should have 1 channel"

        temperature = label #input["temperature"]
        prediction = self.forward(inputs)

        loss = loss_fn(prediction, temperature[:, :, 1:, :])

        return loss

    def predict_forward(self, inputs: torch.Tensor, label: torch.Tensor):
        """

        inputs (=permeability) shape: (batch, 1, length, width)
        label (=temperature) shape: (batch, 1, length, width)

        temperature has starting values at index 0 in dimension 2, rest is zero
        """

        steps = label.shape[2]
        assert  inputs.shape[1] == 1, "Input (=Permeability) should have 1 channel"

        for i in range(1, steps):
            prediction = self.forward(inputs=inputs, label=label)
            for j in range(1, i):
                assert (
                    label[:, :, j, :] == prediction[:, :, j - 1, :]
                ).all()
            label[:, :, i, :] = prediction[:, :, i - 1, :]

        return label

    
def example():
    m = AutoRegressive()
    temp = torch.rand(2, 1, 10, 64)
    perm = torch.rand(2, 1, 10, 64)

    prediction = m(inputs = perm, label = temp)
    loss = nn.MSELoss()(prediction, temp[:, :, 1:])
    print(loss)

    temp_pred_empty = torch.zeros_like(temp)
    temp_pred_empty[:, :, 0, :] = temp[:, :, 0, :]
    prediction = m.predict_forward(inputs = perm, label = temp_pred_empty)

    print(prediction)
    print(prediction.shape)

if __name__ == "__main__":
    example()
# %%
