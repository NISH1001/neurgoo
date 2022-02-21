#!/usr/bin/env python3

import numpy as np

from ._base import AbstractLoss
from .structures import Tensor


class MeanSquaredError(AbstractLoss):
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return 1 / len(actual) * np.sum((actual - predicted) ** 2)

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return predicted - actual


class CrossEntropyLoss(AbstractLoss):
    _zero_clipper = 1e-7

    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return np.mean((-actual * np.log(predicted + self._zero_clipper)).sum(axis=1))

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return -actual / ((predicted + self._zero_clipper) * len(actual))

    # def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
    #     return -(actual - predicted) / (
    #         predicted * (1 - predicted) + self._zero_clipper
    #     )


def main():
    pass


if __name__ == "__main__":
    main()
