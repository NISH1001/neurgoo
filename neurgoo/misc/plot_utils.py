#!/usr/bin/env python3

from typing import Dict, List

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False

from loguru import logger

from .eval import EvalData


def plot_losses(losses):
    if not MATPLOTLIB:
        logger.error("Maplotlib not installed. Halting the plot process!")
        return
    plt.plot(losses)
    plt.show()


def plot_history(
    history: Dict[str, List[EvalData]], plot_type="loss", figure_size=(20, 7)
) -> None:
    """
    This function plots train/val metrics in the same figure.
    """
    if not MATPLOTLIB:
        logger.error("Maplotlib not installed. Halting the plot process!")
        return

    train = history.get("train", [])
    val = history.get("val", [])

    # get epoch data common to both
    t_epochs = list(map(lambda e: e.epoch, train))
    v_epochs = list(map(lambda e: e.epoch, val))
    epochs = set(t_epochs).intersection(v_epochs)

    train = filter(lambda e: e.epoch in epochs, train)
    train = sorted(train, key=lambda e: e.epoch)

    val = filter(lambda e: e.epoch in epochs, val)
    val = sorted(val, key=lambda e: e.epoch)

    plt.figure(figsize=figure_size)
    plt.plot([getattr(data, plot_type) for data in train])
    plt.plot([getattr(data, plot_type) for data in val])
    plt.legend([f"Train {plot_type}", f"Val {plot_type}"])
    plt.xlabel("epoch")
    plt.ylabel(f"{plot_type}")


def main():
    pass


if __name__ == "__main__":
    main()
