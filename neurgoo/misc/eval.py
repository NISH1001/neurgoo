from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..structures import Tensor


def convert_prob_to_label(y):
    return np.argmax(y, axis=1).reshape((len(y), 1))


@dataclass
class EvalData:
    epoch: int
    accuracy: float
    precision: float = -1
    recall: float = -1
    loss: float = np.nan

    @classmethod
    def default_empty(cls) -> EvalData:
        return cls(epoch=-1, accuracy=-1, precision=-1, recall=-1)


class Evaluator:
    """
    A helper class to evaluate predictions.
    """

    def __init__(self, num_classes: int = 10):
        assert isinstance(num_classes, int)
        self.num_classes = num_classes

    def calculate_accuracy_from_labels(
        self, target_label: Sequence[int], predicted_label: Sequence[int]
    ):
        matched = np.sum(target_label == predicted_label)
        return matched / len(target_label)

    def calculate_accuracy(self, target: Tensor, predicted: Tensor) -> float:
        """
        Args:
            target: ``Tensor`` or ``np.ndarray``
                Target y values, one-hot-encoded.
                Of the dimension (Nxm), where N is batch_size, m is num_classes
        """
        return np.sum(np.argmax(predicted, axis=1) == np.argmax(target, axis=1)) / len(
            target
        )

    def calculate_metrics(
        self, target_label: Sequence[int], predicted_label: Sequence[int]
    ):
        # n = len(target_label)
        result = {}
        cm = np.zeros((self.num_classes, self.num_classes))
        precisions = np.zeros(self.num_classes)
        recalls = np.zeros(self.num_classes)
        for t, p in zip(target_label, predicted_label):
            cm[t][p] += 1

        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp

        for i in range(self.num_classes):
            p_denom = tp[i] + fp[i]
            r_denom = tp[i] + fn[i]
            precisions[i] = 0 if p_denom == 0 else tp[i] / p_denom
            recalls[i] = 0 if r_denom == 0 else tp[i] / r_denom

        return cm, precisions, recalls
