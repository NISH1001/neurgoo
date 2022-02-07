#!/usr/bin/env python3

from typing import Optional, Union

import numpy as np

from .._base import AbstractLayer, OptimParam
from ..structures import Shape, Tensor


class Identity(AbstractLayer):
    """
    A no-op layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super()

    def feed_forward(self, X: Tensor) -> Tensor:
        return X


class Linear(AbstractLayer):
    """
    Represents a linear/dense layer:
        output -> input @ w + b

    Args:
        ``num_neurons``: ``int``
            Total number of "Linear" neurons in the layer

        ``in_features``: ``int``
            Expected size of input.

        ``use_bias``: ``bool``
            If set to `False`, we won't consider bias for linear operation

        ``trainable``: ``bool``
            This flag controls whether we need to update
            all `OptimParam` parameters during optimization step.
            In this case, setting to `False` doesn't allow
            the layer's weights and biases to be updated during training.
    """

    def __init__(
        self,
        in_features: int,
        num_neurons: int,
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
        self.in_features = in_features
        self.use_bias = bool(use_bias)
        self.W: OptimParam = OptimParam.default_empty()
        self.b: OptimParam = OptimParam.default_empty()
        self._input_cache = None
        self.initialize()

    def initialize(self) -> None:
        self._initialize_default()

    def _initialize_default(self) -> None:
        self.W.val = np.random.randn(self.in_features, self.num_neurons)
        if self.use_bias:
            self.b.val = np.random.randn(self.num_neurons)

    def feed_forward(self, X: Tensor) -> Tensor:
        """
        Forward pass to the layer which
        performs linear operation using
        available weights and bias params

        Args:
            X: ``Tensor``
                Input data to the layer
                This is of the shape: ``(batch_size, in_features)``

        Returns:
            ``Tensor`` result after the linear operation.
            This is of the shape: ``(batch_Size, num_neurons)``

        Note:
            The input data is cached that will be used for
            backward pass.

            Also, if `use_bias` is set to False, bias will
            not be used in the forward pass.
        """
        # cache the input for backprop
        self._input_cache = X
        z = X @ self.W.val
        if self.use_bias:
            z = z + self.b.val
        return z

    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        raise NotImplementedError()

    @property
    def output_shape(self) -> Shape:
        return (None, self.num_neurons)

    @property
    def input_shape(self) -> Shape:
        return (None, self.in_features)


def main():
    pass


if __name__ == "__main__":
    main()