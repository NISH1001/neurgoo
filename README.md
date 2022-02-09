# neurgoo
Implementation of modular neural networks

# Components

As of the version `0.1.0 has 3 main components to train any neural network for downstream tasks:

## I) models

Models are the skeleton of `neurgoo` where we can add any arbitrary number of layers for training.
All the models are inherited from `neurgoo._base.AbstractModel` base class that has all the concrete abstraction required.

Each model is expected to be composed of **N** number of layers. Each layer is expected to derive from `neurgoo._base.AbstractLayer` base class.

See `neurgoo.models.DefaultNNModel` for default implementation.

## II) losses

Loss components are required to train any model in a supervised manner.

All the loss layers are inherited from `neurgoo._base.AbstractLoss` base class for which `loss` and `gradient` methods are to be implemented.

`neurgoo.losses.MeanSquaredError` is one implementation in this version.

## III) optimizers

Optimizers are the heart of the training where weights/biases and all the "trainable" parameters are to be updated after backpropagation.

For now, optimizers gradient update method (`neurgoo._base.AstractOptimizer.step()` method works on any parameters that are of type `neurgoo._base.OptimParam`

---

## trainers

Once we have the above 3 things -- model, loss, optimizer -- we can use any trainer (see: `neurgoo.trainers`) to train the model.

---

A test implementation of linear regression exists at `tests.test_models.test_linear_regression()` function.

---

# Installation

`python setup.py install`


`pip install -e .`
