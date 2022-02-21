#!/usr/bin/env python3

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Type

import numpy as np

from .misc.eval import Evaluator
from .structures import NULL_TENSOR, Shape, Tensor


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
        self.mode = "train"

    def train_mode(self) -> None:
        self.mode = "train"

    def eval_mode(self) -> None:
        self.mode = "eval"

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


class AbstractLoss(BaseMixin, ABC):
    def __init__(self, name: Optional[str] = None) -> None:
        name = name or self.__classname__

    @abstractmethod
    def loss(self, actual: Tensor, predicted: Tensor) -> Tensor:
        raise NotImplementedError()

    def __call__(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return self.loss(actual, predicted)

    def feed_forward(self, actual: Tensor, predicted: Tensor) -> Tensor:
        return self.loss(actual, predicted)

    @abstractmethod
    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        raise NotImplementedError()


class OptimParam(BaseMixin):
    """
    Represents a parameter type that any optimizer can affect
    for gradient update step.
    """

    def __init__(
        self, val: Optional[Tensor] = None, requires_grad: bool = True
    ) -> None:
        self.val: Tensor = val or NULL_TENSOR
        self.grad: Tensor = NULL_TENSOR
        self.requires_grad: bool = bool(requires_grad)

    @property
    def shape(self) -> Shape:
        return self.val.shape

    @classmethod
    def default_empty(cls) -> OptimParam:
        return cls(np.array([]), requires_grad=True)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        name = self.__classname__
        return f"{name} || requires_grad={self.requires_grad} || val_shape = {self.val.shape} || grad_shape = {self.grad.shape}"


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

    def predict(self, X: Tensor) -> Tensor:
        return self.feed_forward(X)

    def feed_forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer.feed_forward(X)
        return X

    def __getitem__(self, index: int) -> Type[AbstractLayer]:
        return self.layers[index]

    def __call__(self, X: Tensor) -> Tensor:
        return self.feed_forward(X)

    def params(self) -> Tuple[OptimParam]:
        res = []
        for layer in self.layers:
            for var, t in layer.__dict__.items():
                if isinstance(t, OptimParam):
                    res.append(getattr(layer, var))
        return tuple(res)

    def backpropagate(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backpropagate(grad)
        return grad

    def train_mode(self) -> None:
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self) -> None:
        for layer in self.layers:
            layer.eval_mode()

    def __str__(self) -> str:
        name = self.name
        layers_str = "\n".join([str(layer) for layer in self.layers])
        return f"[Model=({name}, {self.__classname__})]\nnum_layers={len(self.layers)}\nLayers=[\n{layers_str}\n]"


class AbstractOptimizer(BaseMixin, ABC):
    def __init__(
        self, params: Tuple[OptimParam], lr: float = 1e-3, debug: bool = False
    ) -> None:
        self._sanity_check_params(params)
        self.params = params

        if not isinstance(lr, float):
            raise TypeError(f"Invalid type for lr. Expected float. Got {type(lr)}")
        self.lr = lr

        self.debug = bool(debug)

    def _sanity_check_params(self, params: Tuple[OptimParam]) -> bool:
        assert params is not None
        if not isinstance(params, tuple):
            raise TypeError(
                f"Invalid type for params. Expected tuple. Got {type(params)}"
            )
        for i, param in enumerate(params):
            if not isinstance(param, OptimParam):
                raise TypeError(
                    f"Invalid type for param at index={i}. Expected type of OptimParam. Got {type(param)}"
                )

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError()


class AbstractModelTrainer(BaseMixin, ABC):
    def __init__(
        self,
        model: Type[AbstractModel],
        loss: Type[AbstractLoss],
        optimizer: Type[AbstractOptimizer],
        evaluator: Evaluator,
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        if not isinstance(model, AbstractModel):
            raise TypeError(
                f"Invalid type for model. Expected any type of AbstractModel. Got {type(model)}"
            )
        self.model = model

        if not isinstance(loss, AbstractLoss):
            raise TypeError(
                f"Invalid type for loss. Expected any type of AbstractLoss. Got {type(loss)}"
            )
        self.loss = loss

        if not isinstance(optimizer, AbstractOptimizer):
            raise TypeError(
                f"Invalid type for optimizer. Expected any type of AbstractOptimizer. Got {type(optimizer)}"
            )
        self.optimizer = optimizer

        self.training_losses = []
        self.costs = []
        self.evaluator = evaluator

    @abstractmethod
    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        X_test: Tensor,
        Y_test: Tensor,
        nepochs: int,
        batch_size: int,
    ) -> Tensor:
        raise NotImplementedError()

    def train(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        X_test: Tensor,
        Y_test: Tensor,
        nepochs: int,
        batch_size: int,
    ) -> Tensor:
        raise self.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            nepochs=nepochs,
            batch_size=batch_size,
        )

    def _shuffle(self, X: Tensor, Y: Tensor) -> Tuple[Tensor]:
        indices = list(range(len(X)))
        random.shuffle(indices)
        return X[indices], Y[indices]


def main():
    pass


if __name__ == "__main__":
    main()
