from typing import *

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_graph(path_to_file: str):
    data = []  # iterations, epochs, losses, lm_losses, mc_losses
    with open(path_to_file, "r") as f:
        for line in f:
            data.append(list(map(float, line.strip().split(","))))

    data = np.array(data)

    fig, ax = plt.subplots()

    ax.set_yscale("log")

    ax.scatter(data[:, 0], data[:, 2], marker=".", color="blue", alpha=0.01)
    ax.scatter(data[:, 0], data[:, 3], marker=".", color="red", alpha=0.01)
    ax.scatter(data[:, 0], data[:, 4], marker=".", color="green", alpha=0.01)

    print(data)

    plt.show()


if __name__ == "__main__":
    plot_loss_graph("log/log-20-12-22-13-00-17.txt")
