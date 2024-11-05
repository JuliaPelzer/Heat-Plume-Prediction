import torch
from torch import nn


class CausalConv2d(nn.Module):
    def __init__(self, level: int = 0) -> None:
        dilation = 2**level
        super().__init__()
        kernel_size_x = 2
        self.padding_x = (kernel_size_x - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(kernel_size_x, 3),
            padding=(self.padding_x, 1),
            dilation=(dilation, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        return x[:, :, : -self.padding_x, :]


class Encoder(nn.Module):
    def __init__(self, levels: int = 1):
        super().__init__()
        self.convs = nn.Sequential(*[CausalConv2d(level=i) for i in range(levels)])

    def forward(self, x):
        return self.convs(x)


# %%
class AutoRegressive(nn.Module):
    def __init__(self, levels: int = 2) -> None:
        super().__init__()
        self.temp_enc = Encoder(levels=levels)
        self.perm_enc = Encoder(levels=levels)
        self.loss = nn.MSELoss()

    def forward(self, input: dict[str, torch.Tensor]):
        """

        temperature shape: (batch, 1, length, width)
        permeability shape: (batch, 1, length, width)
        """
        temperature = input["temperature"]
        permeability = input["permeability"]

        temp_enc = self.temp_enc(temperature[:, :, :-1, :])
        perm_enc = torch.flip(self.perm_enc(torch.flip(permeability, (2,))), (2,))

        temperature = temperature[:, :, 1:, :]
        perm_enc = perm_enc[:, :, 1:, :]

        prediction = nn.functional.tanh(temp_enc + perm_enc).add(1).div(2)

        return prediction

    def get_loss(self, input: dict[str, torch.Tensor]):
        """

        temperature shape: (batch, 1, length, width)
        permeability shape: (batch, 1, length, width)
        """

        temperature = input["temperature"]
        prediction = self.forward(input)

        loss = self.loss(prediction, temperature[:, :, 1:, :])

        return loss

    def predict_forward(self, input: dict[str, torch.Tensor]):
        """

        temperature shape: (batch, 1, length, width)
        permeability shape: (batch, 1, length, width)

        temperature has starting values at index 0 in dimension 2, rest is zero
        """

        steps = input["temperature"].shape[2]

        for i in range(1, steps):
            prediction = self.forward(input)
            for j in range(1, i):
                assert (
                    input["temperature"][:, :, j, :] == prediction[:, :, j - 1, :]
                ).all()
            input["temperature"][:, :, i, :] = prediction[:, :, i - 1, :]

        return input["temperature"]


def example():
    m = AutoRegressive()
    temp = torch.rand(2, 1, 10, 64)
    perm = torch.rand(2, 1, 10, 64)

    loss = m.get_loss({"temperature": temp, "permeability": perm})
    print(loss)
    prediction = m.predict_forward({"temperature": temp, "permeability": perm})
    print(prediction)
    print(prediction.shape)

if __name__ == "__main__":
    example()