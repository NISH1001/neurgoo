#!/usr/bin/env python3

import numpy as np
from loguru import logger

from ._base import AbstractModelTrainer
from .structures import Tensor, TensorArray


class DefaultModelTrainer(AbstractModelTrainer):
    def fit(self, X: Tensor, Y: Tensor, nepochs: int):
        losses = []
        for epoch in range(nepochs):
            predicted = self.model.feed_forward(X)

            loss = self.loss.loss(Y, predicted)
            losses.append(loss)

            current_cost = TensorArray(loss).mean()
            self.costs.append(current_cost)

            if self.debug:
                logger.debug(f"Epoch={epoch} | Cost={current_cost}")

            grad = self.loss.gradient(Y, predicted)
            self.model.backpropagate(grad)
            self.optimizer.step()

        self.training_losses += losses
        return losses


def main():
    pass


if __name__ == "__main__":
    main()
