#!/usr/bin/env python3

import numpy as np

from ._base import AbstractModelTrainer
from .structures import Tensor, TensorArray


class DefaultModelTrainer(AbstractModelTrainer):
    def fit(self, X: Tensor, Y: Tensor, nepochs: int):
        losses = []
        for e in range(nepochs):
            predicted = self.model.feed_forward(X)

            loss = self.loss.loss(Y, predicted)
            losses.append(loss)

            self.costs.append(TensorArray(loss).mean())

            grad = self.loss.gradient(Y, predicted)
            self.model.backpropagate(grad)
            self.optimizer.step()

        self.training_losses += losses
        return losses


def main():
    pass


if __name__ == "__main__":
    main()
