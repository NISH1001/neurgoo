#!/usr/bin/env python3

from typing import Optional, Union

import numpy as np

from ._base import AbstractLayer, OptParam, Tensor


class Linear(AbstractLayer):
    """
    Represents a linear/dense layer:
        output -> input @ w + b

    Args:
        num_neurons: ``int``
            Total number of "Linear" neurons in the layer

        input_shape: ``Union[int, tuple]``
            Expected input shape

    """

    def __init__(
        self,
        num_neurons: int,
        input_shape: Union[tuple] = None,
        use_bias: bool = True,
        trainable: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(trainable=trainable, debug=debug)
        if not isinstance(num_neurons, int):
            raise TypeError(
                f"Invalid typefor num_neurons. Expected int. Got {type(num_neurons)}"
            )
        if num_neurons <= 0:
            raise ValueError(
                f"Invalid value for num_neurons. Expected >0. Got {num_neurons}"
            )

        self.num_neurons = num_neurons
        self.use_bias = bool(use_bias)
        self.input_shape = input_shape
        self.W: OptParam = OptParam.default_empty()
        self.b: OptParam = OptParam.default_empty()
        self._input_cache = None
        self.initialize()

    def initialize(self) -> None:
        self._initialize_default()

    def _initialize_default(self) -> None:
        self.W.val = np.random.randn(self.input_shape[1], self.num_neurons)
        if self.use_bias:
            self.b.val = np.random.randn(self.num_neurons)

    def feed_forward(self, X: Tensor) -> Tensor:
        # cache the input for backprop
        self._input_cache = X
        z = X @ self.W.val
        if self.use_bias:
            z = z + self.b.val
        return z

    def backpropagate(self, dout):
        raise NotImplementedError()

    @property
    def output_shape(self) -> tuple:
        return (None, self.num_neurons)


def main():
    pass


if __name__ == "__main__":
    main()
