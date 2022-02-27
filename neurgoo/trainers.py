#!/usr/bin/env python3

import time
from typing import Dict, List

import numpy as np
from loguru import logger
from tqdm import tqdm

from ._base import AbstractModelTrainer
from .misc.eval import EvalData
from .structures import Tensor


class DefaultModelTrainer(AbstractModelTrainer):
    """
    This is the default implementation of the training loop.

    - loops through each batch for every epoch
    - does forward pass and backprop for each batch
    - evaluates train and val at the end of each epoch
    - evaluates test at the end of training (after final epoch)
    """

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
    ) -> Dict[str, List[EvalData]]:
        """
        Args:
            ``X_train``: ``np.ndarray``
                Training input X
                Shape of (N, m),
                N being number of input instances
                m being number of input features
            ``Y_train``: ``np.ndarray``
                One-hot encoded Target/Ground truths
                Shape of (N, k)
                    - N being number of instances
                    - K being total number of classes for classification
            ``X_val``: ``np.ndarray``
                Validation input data
            ``Y_val``: `np.ndarray`
                Validation target data
            ``X_test``: ``np.ndarray``
                Test input data
            ``Y_test``: ``np.ndarray``
                Test target data
            ``nepochs``: ``int``
                How many epoch to train the whole data?
            ``batch_size``: ``int``
                For each epoch, how many chunk of data
                is used for training?
            ``epoch_shuffle``: ``bool``
                If set, input data (X_train, Y_train) will be shuffled
                at the start of each epoch

        Returns:
            ``Dict[str, List[EvalData]]``
                The dictionary consists of `train`, `val` and `test` keys.
                Basically, it's a dict to track training history.

        Sample dict output:

            .. code-block:: python

                {'test': EvalData(epoch=3, accuracy=0.9802, precision=-1, recall=-1, loss=0.07289549116303999),
                'train': [EvalData(epoch=1, accuracy=0.991, precision=-1, recall=-1, loss=0.04274904622786173),
                        EvalData(epoch=2, accuracy=0.99225, precision=-1, recall=-1, loss=0.037564847851979145),
                        EvalData(epoch=3, accuracy=0.9925333333333334, precision=-1, recall=-1, loss=0.03418293240649179)],
                'val': [EvalData(epoch=1, accuracy=0.9802, precision=-1, recall=-1, loss=0.07474111306150792),
                        EvalData(epoch=2, accuracy=0.9824, precision=-1, recall=-1, loss=0.06985540943935817),
                        EvalData(epoch=3, accuracy=0.9802, precision=-1, recall=-1, loss=0.07289549116303999)]}
        """
        logger.info(f"Training | nepochs={nepochs} | batch_size={batch_size}")
        grand_time = time.time()
        history = dict(train=[], val=[], test=EvalData.default_empty())
        for epoch in range(1, nepochs + 1):
            # This is a very naive "emulation" of pytorch
            # All it does is set the layers trainable and also allows input
            # caching.
            self.model.train_mode()

            if epoch_shuffle:
                X_train, Y_train = self._shuffle(X_train, Y_train)

            start = time.time()
            epoch_costs = []
            for i, k in tqdm(enumerate(range(0, len(X_train), batch_size))):
                x_batch = X_train[k : k + batch_size]
                y_batch = Y_train[k : k + batch_size]

                predicted = self.model.feed_forward(x_batch)

                grad = self.loss.gradient(y_batch, predicted)
                self.model.backpropagate(grad)

                # A very naive implementation to emulate pytorch's optimizer
                # if we don't do this, weights won't be updated!
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

            # Just a naive emulation of pytorch's eval mode
            # This disables training. Also it disables input caching during
            # forward pass as we don't need to do backward pass for eval mode.
            # Save memory drastically!
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

        predicted_test = self.model.predict(X_test)
        test_acc = self.evaluator.calculate_accuracy(Y_test, predicted_test)
        test_cost = self.loss.loss(Y_test, predicted_test)
        history["test"] = EvalData(epoch=epoch, accuracy=test_acc, loss=test_cost)
        train_cost = history["train"][-1].loss
        logger.info(
            f"After training | Epoch={epoch}/{nepochs} | Train loss={train_cost} | Test loss={test_cost} | Train Acc={train_acc} | Test Acc={test_acc} | Delta time={time.time()-grand_time}"
        )

        return history


def main():
    pass


if __name__ == "__main__":
    main()
