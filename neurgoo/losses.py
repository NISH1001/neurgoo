#!/usr/bin/env python3

import numpy as np

from ._base import AbstractLoss
from .structures import Tensor


class MeanSquaredError(AbstractLoss):
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return np.mean((predicted - actual) ** 2) / len(actual)

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return predicted - actual


# class CrossEntropyLoss(AbstractLoss):
#     _zero_clipper = 1e-10

#     def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
#         # to avoid division by zero, log(0) is undefined
#         predicted = np.clip(predicted, self._zero_clipper, 1 - self._zero_clipper)
#         return -actual * np.log(predicted)  # - (1 - actual) * np.log(1 - predicted)

#     def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
#         predicted = np.clip(predicted, self._zero_clipper, 1 - self._zero_clipper)
#         return -(actual / predicted)  # + (1 - actual) / (1 - predicted)


class CrossEntropyLoss(AbstractLoss):
    _zero_clipper = 1e-7

    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return (
            -1
            / len(actual)
            * np.sum(
                actual * np.log(predicted + self._zero_clipper)
                + (1 - actual) * np.log(1 - predicted + self._zero_clipper)
            )
        )

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return -(actual - predicted) / (
            predicted * (1 - predicted) + self._zero_clipper
        )


class HingeLoss(AbstractLoss):
    """
    Note:
        - This should be acted on raw outputs/logits (no softmax!)
        - different of unnormalized logits
        - loss for k-class?
    """

    _zero_clipper = 1e-10

    def loss(self, actual: Tensor, logits: Tensor) -> Tensor:
        """
        Args:
            logits: ``Tensor``
                Unnormalized (unactivated) output from linear layer
        """
        raise NotImplementedError()

    def gradient(self, actual: Tensor, logits: Tensor) -> Tensor:
        raise NotImplementedError()


def main():
    pass


if __name__ == "__main__":
    main()
