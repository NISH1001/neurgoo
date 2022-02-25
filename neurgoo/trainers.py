#!/usr/bin/env python3

import time

import numpy as np
from loguru import logger

from ._base import AbstractModelTrainer
from .layers.activations import ActivationLayer, Softmax
from .losses import CrossEntropyLossWithLogits
from .misc.eval import EvalData
from .structures import Tensor, TensorArray


class DefaultModelTrainer(AbstractModelTrainer):
    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        X_val: Tensor,
        Y_val: Tensor,
        X_test: Tensor,
        Y_test: Tensor,
        nepochs: int,
        batch_size: int = 64,
        epoch_shuffle: bool = True,
    ):
        logger.info(f"Training | nepochs={nepochs} | batch_size={batch_size}")
        losses = []
        grand_time = time.time()
        history = dict(train=[], val=[], test=EvalData.default_empty())
        for epoch in range(1, nepochs + 1):
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

                loss = self.loss.loss(y_batch, predicted)
                epoch_costs.append(loss)

                if self.debug:
                    train_acc_batch = self.evaluator.calculate_accuracy(
                        y_batch, (self.model.predict(x_batch))
                    )
                    logger.debug(
                        f"Epoch={epoch}/{nepochs} | Batch={i} | Batch loss={loss} | Batch Train Acc={train_acc_batch}"
                    )

            self.model.eval_mode()

            current_cost = np.mean(epoch_costs)
            train_acc = self.evaluator.calculate_accuracy(
                Y_train, self.model.predict(X_train)
            )
            history["train"].append(
                EvalData(epoch=epoch, accuracy=train_acc, loss=current_cost)
            )

            predicted_val = self.model.predict(X_val)
            val_loss = self.loss.loss(Y_val, predicted_val)
            val_acc = self.evaluator.calculate_accuracy(Y_val, predicted_val)
            history["val"].append(
                EvalData(epoch=epoch, accuracy=val_acc, loss=val_loss)
            )

            logger.debug(
                f"Epoch={epoch}/{nepochs} | Epoch Train Cost={current_cost} | | Epoch Val Cost={val_loss} | Train Acc={train_acc} | Val Acc={val_acc} | Delta time={time.time()-start}"
            )

        test_acc = self.evaluator.calculate_accuracy(Y_test, self.model.predict(X_test))
        history["test"] = EvalData(epoch=epoch, accuracy=test_acc)
        logger.info(
            f"After training | Epoch={epoch}/{nepochs} | Final cost={losses[-1]} | Train Acc={train_acc} | Test Acc={test_acc} | Delta time={time.time()-grand_time}"
        )

        return losses


class LogitsModelTrainer(AbstractModelTrainer):
    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        X_val: Tensor,
        Y_val: Tensor,
        X_test: Tensor,
        Y_test: Tensor,
        nepochs: int,
        batch_size: int = 64,
        epoch_shuffle: bool = True,
    ):
        if isinstance(self.model.layers[-1], ActivationLayer):
            raise TypeError(
                "To use this trainer, make sure  that last layer of the model shouldn't have any activation."
                + " This trainer only accepts neurgoo.losses.CrossEntropyLossWithLogits!"
                + " Maybe you want to use neurgoo.trainer.DefaultModelTrainer"
            )
        assert isinstance(self.loss, CrossEntropyLossWithLogits)

        logger.info(f"Training | nepochs={nepochs} | batch_size={batch_size}")
        losses = []
        grand_time = time.time()
        history = dict(train=[], val=[], test=EvalData.default_empty())
        for epoch in range(1, nepochs + 1):
            self.model.train_mode()
            if epoch_shuffle:
                X_train, Y_train = self._shuffle(X_train, Y_train)

            start = time.time()
            epoch_costs = []
            for i, k in enumerate(range(0, len(X_train), batch_size)):
                x_batch = X_train[k : k + batch_size]
                y_batch = Y_train[k : k + batch_size]

                logits = self.model.feed_forward(x_batch)

                grad = self.loss.gradient(y_batch, logits)
                self.model.backpropagate(grad)
                self.optimizer.step()

                loss = self.loss.loss(y_batch, logits)
                epoch_costs.append(loss)

                if self.debug:
                    train_acc_batch = self.evaluator.calculate_accuracy(
                        y_batch, Softmax()(self.model.predict(x_batch))
                    )
                    logger.debug(
                        f"Epoch={epoch}/{nepochs} | Batch={i} | Batch loss={loss} | Batch Train Acc={train_acc_batch}"
                    )

            self.model.eval_mode()

            current_cost = np.mean(epoch_costs)
            train_acc = self.evaluator.calculate_accuracy(
                Y_train, Softmax()(self.model.predict(X_train))
            )
            history["train"].append(
                EvalData(epoch=epoch, accuracy=train_acc, loss=current_cost)
            )

            predicted_val = self.model.predict(X_val)
            val_loss = self.loss.loss(Y_val, predicted_val)
            val_acc = self.evaluator.calculate_accuracy(Y_val, Softmax()(predicted_val))
            history["val"].append(
                EvalData(epoch=epoch, accuracy=val_acc, loss=val_loss)
            )

            logger.info(
                f"Epoch={epoch}/{nepochs} | Epoch Train Cost={current_cost} | | Epoch Val Cost={val_loss} | Train Acc={train_acc} | Val Acc={val_acc} | Delta time={time.time()-start}"
            )

        # self.training_losses += losses
        test_acc = self.evaluator.calculate_accuracy(
            Y_test, Softmax()(self.model.predict(X_test))
        )
        history["test"] = EvalData(epoch=epoch, accuracy=test_acc)
        logger.info(
            f"After training | Epoch={epoch}/{nepochs} | Final cost={losses[-1] if losses else None} | Train Acc={train_acc} | Test Acc={test_acc} | Delta time={time.time()-grand_time}"
        )
        return history


def main():
    pass


if __name__ == "__main__":
    main()
