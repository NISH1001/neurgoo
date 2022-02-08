#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Type, Union

import numpy as np

from .structures import Shape, Tensor


class BaseMixin:
    @property
    def __classname__(self) -> str:
        return self.__class__.__name__


class AbstractLayer(BaseMixin, ABC):
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

    def __init__(self, name: Optional[str] = None, debug: bool = False) -> None:
        super().__init__(trainable=False, debug=debug)
        self.name = name or self.layer_name
        # setting to 0 for the sake of "tensor consistency"
        # could have done with None
        self._input_cache = Tensor(0)

    def initialize(self) -> None:
        pass

    @abstractmethod
    def gradient(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError()

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


class AbstractModel(AbstractLayer):
    def __init__(
        self,
        layers: Optional[Sequence[Type[AbstractLayer]]] = None,
        name: Optional[str] = None,
        trainable: bool = True,
        debug: bool = False,
    ):
        self.name = name or self.__classname__
        self.trainable = bool(trainable)
        self.debug = bool(debug)

        layers = list(layers or [])
        self._sanity_check_layers(layers)
        self.layers = layers

    def initialize(self) -> None:
        pass

    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        raise NotImplementedError("Model doesn't support backprop!")

    def _sanity_check_layers(self, layers: Sequence[Type[AbstractLayer]]) -> bool:
        if layers is None:
            return True
        if layers is not None and not isinstance(layers, (list, tuple)):
            raise TypeError(
                f"Invalid type for layers. Expected any of list, tuple. Got {type(layers)}"
            )
        for i, layer in enumerate(layers):
            if not isinstance(layer, AbstractLayer):
                raise TypeError(f"Invalid type for [layer={layer}] at [index={i}]")
        return True

    def add_layer(self, layer: Type[AbstractLayer]) -> Type[AbstractModel]:
        if not isinstance(layer, AbstractLayer):
            raise TypeError(
                f"Invalid type for layer={layer}. Expected a base type of AbstractLayer. Got {type(layer)}"
            )
        self.layers.append(layer)
        return self

    def add_layers(self, layers: Sequence[Type[AbstractLayer]]) -> Type[AbstractModel]:
        self._sanity_check_layers(layers)
        for layer in layers:
            self.add_layer(layer)
        return self

    def fit(self, X: Tensor, Y: Tensor) -> Tensor:
        raise NotImplementedError()

    def predict(self, X: Tensor) -> Tensor:
        return self.feed_forward(X)

    def feed_forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer.feed_forward(X)
        return X

    def __call__(self, X: Tensor) -> Tensor:
        return self.feed_forward(X)

    def __str__(self) -> str:
        name = self.name
        layers_str = "\n".join([str(layer) for layer in self.layers])
        return f"[Model=({name}, {self.__classname__})]\nnum_layers={len(self.layers)}\nLayers=[\n{layers_str}\n]"


def main():
    pass


if __name__ == "__main__":
    main()
