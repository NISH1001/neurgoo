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

    We use this as the gradient is simple when we merge softmax+cross-entropy
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


class HingeLoss(AbstractLoss):
    """
    This is the loss which works on un-activated/normalized values.
    That is: it works on logits.

    So:
        - First, figure out which logits/classes have larger margin (<1) w.r.t the
        actual class
        - Then, compute losses w.r.t those logits.
        - Suppose the data belongs to class **j**.
        - We want $$\hat{y_{j}}$$ to be at least **one larger**
        than any other $$\hat{y_{i}}, i \neq j$$
    """

    def loss(self, actual: Tensor, logits: Tensor) -> Tensor:
        temp = logits - np.sum(actual * logits, axis=1).reshape(-1, 1) + 1
        return np.mean(np.sum(temp * (temp > 0) * (actual != 1), axis=1))

    def gradient(self, actual: Tensor, logits: Tensor) -> Tensor:
        """
        The gradient is computed tentatively as:
            For every low margin (<1) from target class K to other
            classes 'j',
            we accumulate negative gradients to the target class.
            And we add positive gradient to classes 'j'.

            So, final result is a zero-sum.

        """
        temp = logits - np.sum(actual * logits, axis=1).reshape(-1, 1) + 1
        dist = np.maximum(temp, np.zeros_like(temp))
        vals = (dist > 0) * (actual != 1)

        # for target class K
        grad1 = (-1) * np.sum(vals, axis=1).reshape(-1, 1) * actual

        # for other classses where margin is <1 w.r.t K
        grad2 = ((dist > 0) * (actual != 1)).astype(int)

        return (grad1 + grad2) / len(logits)


def main():
    pass


if __name__ == "__main__":
    main()
