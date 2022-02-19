from typing import Sequence

import numpy as np

from ..structures import Tensor


class Evaluator:
    def __init__(self, num_classes: int = 10):
        assert isinstance(num_classes, int)
        self.num_classes = num_classes

    def calculate_accuracy(
        self, target_label: Sequence[int], predicted_label: Sequence[int]
    ):
        matched = np.sum(target_label == predicted_label)
        return matched / len(target_label)

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
