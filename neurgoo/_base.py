#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

Tensor = np.ndarray


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

    # @property
    # def input_shape(self) -> tuple:
    #     return tuple()

    @property
    def output_shape(self) -> tuple:
        return tuple()

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
    def backpropagate(self, dout: Tensor) -> Tensor:
        """
        Back-propagate the gradient

        Args:
            dout: ``Tensor``
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
        return f"Shape: ({self.input_shape}, {self.output_shape}) | trainable: {self.trainable}"


class AbstractActivation(AbstractLayer):
    """
    Defines layer type of "Activation".
    New activations should derive from this.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        name = name or self.layer_name


class OptParam(AbstractLayer):
    def __init__(self, val: Optional[Tensor] = None) -> None:
        self.val = val
        self.grad = Tensor([])

    def initialize(self):
        pass

    def backpropagate(self):
        pass

    def feed_forward(self):
        pass

    @classmethod
    def default_empty(cls) -> OptParam:
        return cls(np.array([]))


def main():
    pass


if __name__ == "__main__":
    main()
