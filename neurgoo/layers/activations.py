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


class Softmax(Activation, ActivationLayer):
    """
    With axis=1, apply softmax for each row
    """

    def __init__(self, theta: float = 1.0, axis: int = 1) -> None:
        self.theta = theta
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        # make X at least 2d
        y = np.atleast_2d(x)

        # find axis
        axis = self.axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

        y = y * float(self.theta)

        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)
        y = np.exp(y)

        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

        p = y / ax_sum

        # flatten if X was 1D
        if len(x.shape) == 1:
            p = p.flatten()

        return p

    def gradient(self, x: Tensor) -> Tensor:
        y = self(x)
        return y * (1 - y)


def main():
    pass


if __name__ == "__main__":
    main()
