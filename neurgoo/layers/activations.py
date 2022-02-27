#!/usr/bin/env python3

import numpy as np

from .._base import ActivationLayer
from ..structures import Tensor


class Sigmoid(ActivationLayer):
    def __call__(self, x: Tensor) -> Tensor:
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x: Tensor) -> Tensor:
        y = self(x)
        return y * (1 - y)


class ReLU(ActivationLayer):
    def __call__(self, x: Tensor) -> Tensor:
        return np.where(x >= 0, x, 0)

    def gradient(self, x: Tensor) -> Tensor:
        return np.where(x >= 0, 1, 0)


class LeakyReLU(ActivationLayer):
    def __init__(self, leak: float = 1e-5, name=None, debug: bool = False) -> None:
        super().__init__()

        if not isinstance(leak, float):
            raise TypeError(
                f"Invalid type for leak={leak}. Expected float. Got {type(leak)}"
            )
        self.leak = leak

    def __call__(self, x: Tensor) -> Tensor:
        return np.where(x > 0, x, x * self.leak)

    def gradient(self, x: Tensor) -> Tensor:
        return np.where(x > 0, 1, self.leak)


class Softmax(ActivationLayer):
    _zero_clipper = 1e-13

    def __call__(self, x: Tensor) -> Tensor:
        # for stability, perform this substraction
        exps = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def gradient(self, x: Tensor) -> Tensor:
        y = self(x + self._zero_clipper)
        return y * (1 - y)


def main():
    pass


if __name__ == "__main__":
    main()
