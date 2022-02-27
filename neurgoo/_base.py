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

    Two main attributes of any layer is to perform:
        - forward pass (see `feed_forward(...)`)
        - back propagation (see `backpropagate(...)`)

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
        self.trainable = True
        self.mode = "train"

    def eval_mode(self) -> None:
        self.trainable = False
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


class ActivationLayer(AbstractLayer):
    """
    This class represents an activation layer.

    Some of the implementation can be found at `neurgoo.layers.activations`:
        - `neurgoo.layers.activations.Sigmoid`
        - `neurgoo.layers.activations.ReLU`
        - `neurgoo.layers.activations.LeakyReLU`
        - `neurgoo.layers.activations.Softmax`

    Each activation layer has 2 primary attributes which should be implemented:
        - `__call__(...)` method which acts as a functor, and allows us to do object
        calls
        - `gradient(...)` which computes the gradient w.r.t input to the
        function. Down the line this is used in `backpropagate(...)` method

    """

    def __init__(self):
        self._input_cache = NULL_TENSOR.copy()
        self.mode = "train"

    def initialize(self):
        pass

    def feed_forward(self, x):
        if self.mode == "train":
            self._input_cache = x
            self.trainable = True
        elif self.mode == "eval":
            self.trainable = False
        return self(x)

    def __call__(self, x):
        raise NotImplementedError()

    def backpropagate(self, grad_accum: Tensor) -> Tensor:
        return grad_accum * self.gradient(self._input_cache)

    def gradient(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.__classname__} || Attrs => {self.__dict__}"


class AbstractLoss(BaseMixin, ABC):
    """
    This class represents loss component of the neural network system.

    Any implementation of the loss should have 2 main attributes:
        - `loss`, which computes the loss when target and predictions are
        given
        - `gradient` which computes the gradient required for
        `backpropagate(...)` methods

    Current implementations:
        - `neurgoo.losses.MeanSquaredError`
        - `neurgoo.losses.BinaryCrossEntropyLoss`
        - `neurgoo.losses.CrossEntropyLossWithLogits`
        - `neurgoo.losses.HingeLoss`
    """

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

    This is a very naive-implementation to make sure we have loose
    segregation between layer, backpropagation and optimizer.

    Currently we have weights and biases as two `OptimParam` object
    at `neurgoo.layers.Linear`.

    Also, when we do `model.params()`, we are basically
    getting references to layer's OptimParam variables.
    These params are passed to any optimizer to perform gradient update operation.

    Attributes:
        `val`: ``np.ndarray` aliased as `Tensor`
            This stores actual array
        ``grad``: ``Tensor``
            This stores delta value to be used for updating `val`
            in the optimizer
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
    """
    This is an abstraction for a collection of layers.
    In this we can:
        - add any number of layers
        - do forward pass (calls feed_forward method of each layer
        iteratively)
        - do backward pass (calls backpropagate method of each layer
        iteratively)

    Note:
        1) `eval_mode()` disables any trainable param
            and also avoids caching of input. This is for memory optimization
            as we don't have to store any input cache as we don't do backpropagate
            during evaluation only mode.
        2) `train_mode()` enables the training and input cache

    See `neurgoo.models.DefaultNNModel` for current implementation.
    """

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
        """
        Return a collection of all the ``OptimParam``
        object stored in any layer.
        """
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
        self.trainable = True
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self) -> None:
        self.trainable = False
        for layer in self.layers:
            layer.eval_mode()

    def __str__(self) -> str:
        name = self.name
        layers_str = "\n".join([str(layer) for layer in self.layers])
        return f"[Model=({name}, {self.__classname__})]\nnum_layers={len(self.layers)}\nLayers=[\n{layers_str}\n]"


class AbstractOptimizer(BaseMixin, ABC):
    """
    For any subsequent optimizer implementation, we should implement their
    `step(...)` method where we access trainable params and update  their
    values accordingly.

    See `neurgoo.optimizers.SGD` for current implementation.
    """

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
    """
    This component encapsulates all the main training loop,
    through `fit(...)` method.

    See `neurgoo.trainers.DefaultModelTrainer` class for current implementation.
    """

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

        if not isinstance(evaluator, Evaluator):
            raise TypeError(
                f"Invalid type for evaluator. Expected type of Evaluator. Got {type(evaluator)}"
            )
        self.evaluator = evaluator

        self.training_losses = []
        self.costs = []

    @abstractmethod
    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        X_val: Tensor,
        Y_val: Tensor,
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
        X_val: Tensor,
        Y_val: Tensor,
        X_test: Tensor,
        Y_test: Tensor,
        nepochs: int,
        batch_size: int,
    ) -> Tensor:
        raise self.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            nepochs=nepochs,
            batch_size=batch_size,
        )

    def _shuffle(self, X: Tensor, Y: Tensor) -> Tuple[Tensor]:
        """
        Randomly shuffles X and Y
        """
        indices = list(range(len(X)))
        random.shuffle(indices)
        return X[indices], Y[indices]


def main():
    pass


if __name__ == "__main__":
    main()
