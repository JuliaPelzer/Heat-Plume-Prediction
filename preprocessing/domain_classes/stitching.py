from dataclasses import dataclass

from torch import maximum


@dataclass
class Stitching:
    method: str
    background_temperature: float

    def __call__(self, current_value: float, additional_value: float):
        if self.method == "max": # elementwise maximum
            return maximum(current_value, additional_value)
        elif self.method == "add":
            if current_value == self.background_temperature:
                return additional_value
            else:
                return current_value + additional_value - self.background_temperature

