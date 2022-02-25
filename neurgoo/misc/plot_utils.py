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
    plt.figure(figsize=figure_size)
    plt.plot([getattr(data, plot_type) for data in train])
    plt.plot([getattr(data, plot_type) for data in val])
    plt.legend([f"Train {plot_type}", f"Val {plot_type}"])
    plt.ylabel(f"{plot_type}")
    plt.xlabel("epoch")


def main():
    pass


if __name__ == "__main__":
    main()
