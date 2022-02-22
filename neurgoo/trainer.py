#!/usr/bin/env python3

import time

import numpy as np
from loguru import logger

from ._base import AbstractModelTrainer
from .misc import eval as neuroeval
from .structures import Tensor, TensorArray


class DefaultModelTrainer(AbstractModelTrainer):
    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        X_test: Tensor,
        Y_test: Tensor,
        nepochs: int,
        batch_size: int = 64,
        epoch_shuffle: bool = True,
    ):
        logger.info(f"Training | nepochs={nepochs} | batch_size={batch_size}")
        losses = []
        for epoch in range(nepochs):
            self.model.train_mode()
            if epoch_shuffle:
                X_train, Y_train = self._shuffle(X_train, Y_train)

            start = time.time()
            epoch_costs = []
            for i, k in enumerate(range(0, len(X_train), batch_size)):
                x_batch = X_train[k : k + batch_size]
                y_batch = Y_train[k : k + batch_size]

                predicted = self.model.feed_forward(x_batch)

                grad = self.loss.gradient(y_batch, predicted)
                self.model.backpropagate(grad)
                self.optimizer.step()

                # if epoch > 10:
                #     self.optimizer.lr = 0.01

                loss = self.loss.loss(y_batch, predicted)
                epoch_costs.append(loss)

                # if self.debug:
                #     logger.debug(f"Epoch={epoch} | Batch={i} | Batch Cost={cost}")

                test_acc = (
                    np.sum(
                        np.argmax(
                            self.model.predict(X_test),
                            axis=1,
                        )
                        == np.argmax(Y_test, axis=1)
                    )
                    / len(X_test)
                )
                print(test_acc)

            self.model.eval_mode()
            current_cost = np.mean(epoch_costs)
            self.costs.append(current_cost)

            # train_labels = neuroeval.convert_prob_to_label(Y_train)
            # train_predicted_labels = neuroeval.convert_prob_to_label(
            #     self.model.feed_forward(X_train)
            # )

            train_acc = (
                np.sum(
                    np.argmax(
                        self.model.predict(X_train.reshape(X_train.shape[0], -1)),
                        axis=1,
                    )
                    == np.argmax(Y_train, axis=1)
                )
                / X_train.shape[0]
            )
            # train_acc = self.evaluator.calculate_accuracy(
            #     train_labels, train_predicted_labels
            # )

            # test_labels = neuroeval.convert_prob_to_label(Y_test)
            # test_predicted_labels = neuroeval.convert_prob_to_label(
            #     self.model.feed_forward(X_test)
            # )
            # test_acc = self.evaluator.calculate_accuracy(
            #     test_labels, test_predicted_labels
            # )
            test_acc = (
                np.sum(
                    np.argmax(
                        self.model.predict(X_test.reshape(X_test.shape[0], -1)), axis=1
                    )
                    == np.argmax(Y_test, axis=1)
                )
                / X_test.shape[0]
            )
            if self.debug:
                logger.debug(
                    f"Epoch={epoch} | Epoch Cost={current_cost} | Train Acc={train_acc} | Test Acc={test_acc} | Delta time={time.time()-start}"
                )

        # self.training_losses += losses
        return losses


def main():
    pass


if __name__ == "__main__":
    main()
