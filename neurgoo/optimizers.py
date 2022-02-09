#!/usr/bin/env python3

from typing import Tuple

from ._base import AbstractOptimizer, OptimParam


class SGD(AbstractOptimizer):
    def __init__(
        self, params: Tuple[OptimParam], lr: float = 1e-3, debug: bool = False
    ) -> None:
        super().__init__(params=params, debug=debug)

        if not isinstance(lr, float):
            raise TypeError(f"Invalid type for lr. Expected float. Got {type(lr)}")
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.val = param.val - self.lr * param.grad


def main():
    pass


if __name__ == "__main__":
    main()
