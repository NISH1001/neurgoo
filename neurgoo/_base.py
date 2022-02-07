#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .structures import Shape, Tensor


class AbstractLayer(ABC):
    """
    This represent an abstract layer from which
    downstream layer classes are implemented

    Note:
        Layers could be:
            - neuron layer
            - activation function
            - neuron
            - loss
        We assume everything is a layer
    """

    def __init__(self, trainable: bool = True, debug: bool = False) -> None:
        self._trainable = bool(trainable)
        self.debug = bool(debug)

    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @property
    def output_shape(self) -> Shape:
        return Shape()

    @property
    def input_shape(self) -> Shape:
        return Shape()

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, val: bool) -> None:
        assert isinstance(val, (bool, int))
        self._trainable = bool(val)

    @abstractmethod
    def feed_forward(self, X: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError()

    def __call__(self, X: Tensor, **kwargs) -> Tensor:
        return self.feed_forward(X, **kwargs)

    @abstractmethod
    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        """
        Back-propagate the gradient

        Args:
            grad_accum: ``Tensor``
                Accumulated gradient wrt the output of the layer.

        Returns:
            ``Tensor`` gradient wrt input to the layer.
        """
        raise NotImplementedError()

    @property
    def num_params(self) -> int:
        return 0

    @property
    def __classname__(self) -> str:
        return self.__class__.__name__

    @property
    def layer_name(self) -> str:
        return self.__classname__

    def __str__(self) -> str:
        return f"{self.__classname__} || Shape: ({self.input_shape}, {self.output_shape}) || trainable: {self.trainable}"


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class ActivationLayer(AbstractLayer):
    """
    Defines layer type of "Activation".
    New activations should derive from this.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.layer_name
        self._input_cache = Tensor(0)

    def initialize(self) -> None:
        pass

    def feed_forward(self, x: Tensor) -> Tensor:
        self._input_cache = x
        return self(x)

    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        return grad_accum * self.gradient(self._input_cache)

    def __str__(self) -> str:
        return f"{self.__classname__} || Attrs => {self.__dict__}"


class LossLayer(AbstractLayer):
    def __init__(self, name: Optional[str] = None) -> None:
        name = name or self.layer_name


class OptimParam:
    """
    Represents a parameter type that any optimizer can affect
    for gradient update step.
    """

    def __init__(
        self, val: Optional[Tensor] = None, requires_grad: bool = True
    ) -> None:
        self.val: Tensor = val or Tensor(0)
        self.grad: Tensor = Tensor(0)
        self.requires_grad: bool = bool(requires_grad)

    @property
    def shape(self) -> Shape:
        return self.val.shape

    @classmethod
    def default_empty(cls) -> OptimParam:
        return cls(np.array([]), requires_grad=True)


def main():
    pass


if __name__ == "__main__":
    main()
