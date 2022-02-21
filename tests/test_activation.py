#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np
from loguru import logger

from neurgoo._base import Activation, ActivationLayer
from neurgoo.layers.activations import Sigmoid, Softmax


def main():
    # X = np.random.randn(42, 12)
    X = np.random.randn(5, 3)
    print(X)

    logger.debug(f"Input shape => {X.shape}")

    # activation = Sigmoid()
    activation = Softmax()
    y = activation.feed_forward(X)
    logger.debug(f"Output shape => {y.shape}")
    logger.info(activation)
    print(isinstance(activation, Activation))
    print(y)
    print(np.sum(y, axis=1))


if __name__ == "__main__":
    main()
