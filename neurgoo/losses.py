#!/usr/bin/env python3

import numpy as np

from ._base import AbstractLoss
from .structures import Tensor


class MeanSquaredError(AbstractLoss):
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return np.mean((predicted - actual) ** 2) / 2

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return predicted - actual


class CrossEntropyLoss(AbstractLoss):
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        # to avoid division by zero
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return -actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        # to avoid division by zero
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return -(actual / predicted) + (1 - actual) / (1 - predicted)


def main():
    pass


if __name__ == "__main__":
    main()
