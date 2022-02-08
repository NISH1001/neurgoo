#!/usr/bin/env python3

from ._base import AbstractModel
from .structures import Tensor


class DefaultNNModel(AbstractModel):
    def fit(self, X: Tensor, Y: Tensor, nepochs: int):
        raise NotImplementedError()


def main():
    pass


if __name__ == "__main__":
    main()
