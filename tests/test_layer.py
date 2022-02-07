#!/usr/bin/env python3

import sys

sys.path.append("../neurgoo/")

import numpy as np

from neurgoo.api import Linear


def main():
    X = np.random.randn(42, 12)
    dense = Linear(num_neurons=3, in_features=12)
    print(dense)
    print(dense.layer_name)
    out = dense.feed_forward(X)
    print(out.shape)


if __name__ == "__main__":
    main()
