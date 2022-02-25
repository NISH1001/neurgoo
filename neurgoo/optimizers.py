#!/usr/bin/env python3

from typing import Tuple

from ._base import AbstractOptimizer, OptimParam


class SGD(AbstractOptimizer):
    def step(self) -> None:
        for param in self.params:
            param.val = param.val - self.lr * param.grad


def main():
    pass


if __name__ == "__main__":
    main()
