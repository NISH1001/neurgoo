#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np
from loguru import logger

from neurgoo.api import Linear
from neurgoo.layers.activations import Sigmoid


def main():
    X = np.random.randn(42, 12)

    logger.debug("-" * 10)
    logger.debug("Forward pass test!")
    dense = Linear(num_neurons=3, in_features=12, debug=True)
    logger.debug(dense)
    logger.debug(f"W shape => {dense.W.shape}")
    logger.debug(f"b shape => {dense.b.shape}")
    logger.debug(f"Initial W Grad => {dense.W.grad}")
    out = dense.feed_forward(X)
    logger.debug(f"Out shape => {out.shape}")
    logger.debug("-" * 10)

    logger.debug("-" * 10)
    logger.debug("Backward pass test!")
    grad = np.random.randn(42, 3)
    grad_accum = dense.backpropagate(grad)
    logger.debug(f"W Grad shape => {dense.W.grad.shape}")
    logger.debug(f"b Grad shape => {dense.b.grad.shape}")
    logger.debug(f"New grad_accum shape => {grad_accum.shape}")
    logger.debug("-" * 10)

    wnew = dense.W.val - dense.W.grad


if __name__ == "__main__":
    main()
