#!/usr/bin/env python3

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False

from loguru import logger


def plot_losses(losses):
    if not MATPLOTLIB:
        logger.error("Maplotlib not installed. Halting the plot process!")
        return
    plt.plot(losses)
    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
