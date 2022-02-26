#!/usr/bin/env python3


from __future__ import annotations

from typing import Optional, Union

import numpy as np
from loguru import logger

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
        self._input_cache = Tensor(0)
        self.mode = "train"
        self.initialize()

    def initialize(self) -> Linear:
        return self.initialize_uniform()

    def initialize_gaussian(self, variance: float = 1.0) -> Linear:
        self.W.val = np.random.normal(
            loc=0, scale=variance ** 0.5, size=(self.in_features, self.num_neurons)
        )
        # self.W.val = np.random.randn(self.in_features, self.num_neurons) * (
        #     variance ** 0.5
        # )
        self.b.val = np.zeros((1, self.num_neurons))
        return self

    def initialize_xavier(self) -> Linear:
        """
        Xavier initialization.

        Note:
            Var(W) = 2/(fan_in + fan_out)

            where,
                fan_in = number of input features
                fan_out = number of output features (neurons)
        """
        variance = 2 / (self.in_features + self.num_neurons)
        self.W.val = np.random.normal(
            loc=0, scale=variance ** 0.5, size=(self.in_features, self.num_neurons)
        )
        # self.W.val = np.random.randn(self.in_features, self.num_neurons) * (
        #     variance ** 0.5
        # )
        self.b.val = np.zeros((1, self.num_neurons))
        return self

    def initialize_uniform(self) -> Linear:
        """
        This only has fan_in
        """
        limit = 1 / np.sqrt(self.in_features)
        self.W.val = np.random.uniform(
            -limit, limit, (self.in_features, self.num_neurons)
        )
        self.b.val = np.zeros((1, self.num_neurons))
        return self

    def initialize_random(self) -> Linear:
        self.W.val = np.random.rand((self.in_features, self.num_neurons))
        self.b.val = np.zeros((1, self.num_neurons))
        return self

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
        if self.mode == "train":
            self._input_cache = X
        z = X @ self.W.val

        # this improves performance also
        # instead of adding "zero" biases to z
        if self.use_bias:
            z = z + self.b.val
        return z

    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        """
        Backward pass gradient compute.

        Args:
            grad_accum: ``Tensor``
                Incoming accumulated gradient from upstream layer
                This actually represents the "accumulated" gradient
                in the chain rule till this layer from the last.

        In each backprop, we need to compute 2 things:
            - find gradient of input grad_accum wrt weights (the main thing)
            - find gradient of input grad_accum wrt input X to the layer

        """
        if self.debug:
            logger.debug(f"Input cache shape => {self._input_cache.shape}")

        if self.trainable:
            # Should be of the same shape as W
            self.W.grad = self._input_cache.T @ grad_accum
            self.b.grad = np.sum(grad_accum, axis=0, keepdims=True)
            # self.b.grad = np.ones((1, grad_accum.shape[0])) @ grad_accum

        # this will be used as a new "grad_accum"
        # in the previous layer  (n-1)
        # wrt input
        # See: how backprop works!

        # debug
        # alpha = 0.001
        # self.W.val = self.W.val - alpha * self.W.grad
        # self.b.val = self.b.val - alpha * self.b.grad
        return grad_accum @ self.W.val.T

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
