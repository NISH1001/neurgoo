#!/usr/bin/env python3

import numpy as np

from ._base import AbstractLoss
from .layers.activations import Softmax
from .structures import Tensor


class MeanSquaredError(AbstractLoss):
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return 1 / len(actual) * np.sum((actual - predicted) ** 2)

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return predicted - actual


class BinaryCrossEntropyLoss(AbstractLoss):
    """
    This is the implementation of binary-cross entropy loss.
    This assumes that there are only two-classes.
    That is: while we need to minimize loss for same class (say 1),
    we need to maximize loss for another (say 0)
    """

    _zero_clipper = 1e-15

    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        zc = self._zero_clipper
        predicted = predicted + zc
        return (
            -1
            / len(actual)
            * np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
        )
        # return np.mean((-actual * np.log(predicted + self._zero_clipper)).sum(axis=1))

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        zc = self._zero_clipper
        return -(actual / (predicted + zc)) + (1 - actual) / (1 - predicted + zc)


class CrossEntropyLossWithLogits(AbstractLoss):
    """
    This is a cross-entropy loss for k-class classification.
    Actually, this loss takes in raw logits (output from linear layer),
    and then applies softmax, and finally computes loss.

    We use this as the gradient flow is simple when we merge softmax+cross-entropy
    into single loss.

    Note:
        - This loss is only used while training using trainer
        `neurgoo.trainers.LogitsModelTrainer`
        - The model shouldn't have any activation layer at the last layer.
    """

    _zero_clipper = 1e-15

    def loss(self, actual: Tensor, logits: Tensor) -> Tensor:
        zc = self._zero_clipper
        predicted = Softmax()(logits) + zc
        return np.mean((-actual * np.log(predicted)).sum(axis=1))

    def gradient(self, actual: Tensor, logits: Tensor) -> Tensor:
        return Softmax()(logits) - actual


def main():
    pass


if __name__ == "__main__":
    main()
