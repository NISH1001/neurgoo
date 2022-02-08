#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np
from loguru import logger

from neurgoo.api import Linear, Sigmoid
from neurgoo.layers.activations import ReLU
from neurgoo.models import DefaultNNModel
from neurgoo.optimizers import SGD


def main():
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


if __name__ == "__main__":
    main()
