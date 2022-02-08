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


def main():
    pass


if __name__ == "__main__":
    main()
