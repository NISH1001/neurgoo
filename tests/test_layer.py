#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np

from neurgoo.api import Linear


def main():
    X = np.random.randn(42, 12)

    print("-" * 10)
    print("Forward pass test!")
    dense = Linear(num_neurons=3, in_features=12)
    print(dense)
    out = dense.feed_forward(X)
    print(out.shape)
    print(f"Initial W Grad => {dense.W.grad}")
    print("-" * 10)

    print("-" * 10)
    print("Backward pass test!")
    grad = np.random.randn(*dense.W.shape)
    grad_accum = dense.backpropagate(grad)
    print(f"Resultant W Grad shape => {dense.W.grad.shape}")


if __name__ == "__main__":
    main()
