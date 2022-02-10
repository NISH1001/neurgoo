#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np
from loguru import logger

from neurgoo.api import Linear, Sigmoid
from neurgoo.layers.activations import ReLU, Softmax
from neurgoo.losses import CrossEntropyLoss, MeanSquaredError
from neurgoo.misc import plot_utils
from neurgoo.models import DefaultNNModel
from neurgoo.optimizers import SGD
from neurgoo.trainer import DefaultModelTrainer


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

    # import matplotlib.pyplot as plt

    # plt.scatter(X[:, 0], X[:, 1], c=["red" if c else "blue" for c in Y.ravel()])
    # plt.show()
    # return

    print(X.shape, Y.shape)
    model = DefaultNNModel()
    model.add_layer(Linear(num_neurons=1, in_features=X.shape[1]))
    model.add_layer(Sigmoid())

    params = model.params()
    optimizer = SGD(params=params, lr=0.001)

    loss = CrossEntropyLoss()
    # loss = MeanSquaredError()
    trainer = DefaultModelTrainer(model=model, optimizer=optimizer, loss=loss)

    errors = trainer.fit(X, Y, nepochs=50)
    # print(errors[0])
    plot_utils.plot_losses(trainer.costs)
    # print(trainer.costs[:30])


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
