#!/usr/bin/env python3

import numpy as np

from .._base import Activation, ActivationLayer
from ..structures import Tensor


class Sigmoid(Activation, ActivationLayer):
    def __call__(self, x: Tensor) -> Tensor:
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x: Tensor) -> Tensor:
        y = self(x)
        return y * (1 - y)


class ReLU(Activation, ActivationLayer):
    def __call__(self, x: Tensor) -> Tensor:
        return np.where(x >= 0, x, 0)

    def gradient(self, x: Tensor) -> Tensor:
        return np.where(x >= 0, 1, 0)


class LeakyReLU(Activation, ActivationLayer):
    def __init__(self, leak: float = 1e-5, name=None, debug: bool = False) -> None:
        super().__init__(name=name, debug=debug)

        if not isinstance(leak, float):
            raise TypeError(
                f"Invalid type for leak={leak}. Expected float. Got {type(leak)}"
            )
        self.leak = leak

    def __call__(self, x: Tensor) -> Tensor:
        return np.where(x > 0, x, self.leak)

    def gradient(self, x: Tensor) -> Tensor:
        return np.where(x > 0, 1, self.leak)


def main():
    pass


if __name__ == "__main__":
    main()
