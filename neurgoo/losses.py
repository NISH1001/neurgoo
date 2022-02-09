#!/usr/bin/env python3

import numpy as np

from ._base import AbstractLoss
from .structures import Tensor


class MeanSquaredError(AbstractLoss):
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return np.mean((predicted - actual) ** 2) / 2

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return predicted - actual


def main():
    pass


if __name__ == "__main__":
    main()
