#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from neurgoo.api import (
    SGD,
    CrossEntropyLoss,
    DefaultModelTrainer,
    DefaultNNModel,
    Linear,
    MeanSquaredError,
    ReLU,
    Sigmoid,
    Softmax,
)
from neurgoo.misc import eval as evaltools
from neurgoo.misc import plot_utils


def test_linear_regression():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 1.0],
            [5.0, 4.0],
            [2.0, 3.0],
            [-1.5, 2.1],
            [0.5, -0.3],
        ]
    )

    real_w = np.array([[1.0], [1.0]])
    real_b = np.array([[0.5]])

    Y = X @ real_w + real_b

    # NN with single neuron
    # no activation
    model = DefaultNNModel()
    model.add_layer(Linear(num_neurons=1, in_features=X.shape[1]))

    params = model.params()
    optimizer = SGD(params=params, lr=0.01)

    loss = MeanSquaredError()
    trainer = DefaultModelTrainer(model=model, optimizer=optimizer, loss=loss)

    errors = trainer.fit(X, Y, nepochs=1000)

    w = model.layers[0].W.val
    b = model.layers[0].b.val

    print(w)
    print(b)

    # Shape of the parameters
    assert w.shape == (2, 1)
    assert b.shape == (1, 1)

    # Value of the parameters
    np.testing.assert_array_almost_equal(w, real_w)
    np.testing.assert_array_almost_equal(b, real_b)

    # Final training error should be zero
    np.testing.assert_almost_equal(errors[-1], 0.0)


def test_logistic_regression():
    # hyperplane is y=x
    N = 25
    X, Y = [], []
    for x in range(-N, N):
        for y in range(-N, N):
            if x < y:
                X.append((x, y))
                Y.append([1])
            if x > y:
                X.append((x, y))
                Y.append([0])

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
    logger.debug(f"X_train shape => {X_train.shape}")
    logger.debug(f"X_test shape => {X_test.shape}")

    # import matplotlib.pyplot as plt

    # plt.scatter(X[:, 0], X[:, 1], c=["red" if c else "blue" for c in Y.ravel()])
    # plt.show()
    # return

    model = DefaultNNModel()
    model.add_layer(Linear(num_neurons=1, in_features=X.shape[1]))
    model.add_layer(Sigmoid())

    params = model.params()
    optimizer = SGD(params=params, lr=0.001)

    loss = CrossEntropyLoss()
    # loss = MeanSquaredError()
    trainer = DefaultModelTrainer(
        model=model, optimizer=optimizer, loss=loss, debug=True
    )

    errors = trainer.fit(X_train, Y_train, nepochs=50)
    # print(errors[0])
    # plot_utils.plot_losses(trainer.costs)
    # print(trainer.costs[:30])

    print("=== EVAL === >")
    predictions = model.predict(X_test)
    # for binary classification, it's fine to do this!
    predictions = predictions.reshape(-1)
    logger.debug(predictions[:10])
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    predictions = predictions.astype(int)
    logger.debug(f"predictions shape => {predictions.shape}")
    logger.debug(predictions[:10])

    gts = Y_test.reshape(-1).astype(int)
    evaluator = evaltools.Evaluator(num_classes=2)
    cm, p, r = evaluator.calculate_metrics(gts, predictions)
    accuracy = evaluator.calculate_accuracy(gts, predictions)

    logger.debug(f"Confusion Matrix => \n{cm}")
    logger.debug(f"Precision : {p} | Recall: {r} | Accuracy: {accuracy}")
    print("<=== EVAL ===")


def test():
    model = DefaultNNModel()

    model.add_layer(Linear(num_neurons=3, in_features=12))
    model.add_layer(Sigmoid())
    model.add_layer(Linear(num_neurons=5, in_features=3))
    model.add_layer(ReLU())
    print(model)

    params = model.params()
    print(params[0].val)

    optimizer = SGD(params=params)

    X = np.random.randn(42, 12)
    Y = model.feed_forward(X)
    print(Y.shape)

    print(params[0].val)

    optimizer.step()


def main():
    # test_linear_regression()
    test_logistic_regression()


if __name__ == "__main__":
    main()
