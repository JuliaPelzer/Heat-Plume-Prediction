def call(self, input_dict):
    # input has shape (batch, height, width, channels), channels should be (2,2)
    distortions = self.model(input_dict["fields"])
    batch, height, width, channels = distortions.shape
    deltas = ops.reshape(distortions, (*distortions.shape[:-1], 2, 2))
    coordinates = ops.zeros((batch, height, width, 2))
    coordinates += ops.cumsum(deltas[..., 1, :], axis=-2)
    coordinates += ops.cumsum(deltas[..., 0, :], axis=-3)
    for index, (oy, ox) in enumerate(input_dict["pump_indices"]):
        coordinates[index, ...] -= coordinates[index, oy, ox, :].clone()

    self.add_loss(self.oob_loss(coordinates) * self.oob_weight)

    self.add_loss(self.ortho_loss(deltas) * self.ortho_weight)

    # TODO: add monotonicity loss back in !!!!
    # TODO: !!!!!!!

    return coordinates
