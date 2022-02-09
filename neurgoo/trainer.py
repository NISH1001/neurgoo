#!/usr/bin/env python3

from ._base import AbstractModelTrainer
from .structures import Tensor


class DefaultModelTrainer(AbstractModelTrainer):
    def fit(self, X: Tensor, Y: Tensor, nepochs: int):
        losses = []
        for e in range(nepochs):
            predicted = self.model.feed_forward(X)

            loss = self.loss.loss(Y, predicted)
            losses.append(loss)

            grad = self.loss.gradient(Y, predicted)
            self.model.backpropagate(grad)
            self.optimizer.step()

        self.training_losses += losses
        return losses


def main():
    pass


if __name__ == "__main__":
    main()
