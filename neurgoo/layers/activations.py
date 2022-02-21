#!/usr/bin/env python3

from typing import Optional

import numpy as np

from .._base import AbstractLayer, BaseMixin
from ..structures import Tensor


class ActivationLayer(AbstractLayer):
    def __init__(self):
        self._input_cache = Tensor(0)

    def initialize(self):
        pass

    def feed_forward(self, x):
        if self.mode == "train":
            self._input_cache = x
        return self(x)

    def __call__(self, x):
        raise NotImplementedError()

    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        return grad_accum * self.gradient(self._input_cache)

    def gradient(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.__classname__} || Attrs => {self.__dict__}"


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
        return np.where(x > 0, x, self.leak)

    def gradient(self, x: Tensor) -> Tensor:
        return np.where(x > 0, 1, self.leak)


class Softmax(ActivationLayer):
    _zero_clipper = 1e-7

    def __call__(self, x: Tensor) -> Tensor:
        sx = x - np.max(x, axis=1).reshape(-1, 1)
        exps = np.exp(sx)
        y = exps / np.sum(exps, axis=1).reshape(-1, 1)
        return y

    def gradient(self, x: Tensor) -> Tensor:
        y = self(x)
        return y * (1 - y)


def main():
    pass


if __name__ == "__main__":
    main()
