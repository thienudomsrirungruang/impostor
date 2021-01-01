from typing import *

import matplotlib.pyplot as plt
import numpy as np


def moving_avg(x, y, num_avg: int):
    xnew = np.convolve(x, np.ones(num_avg), "valid") / num_avg
    ynew = np.convolve(y, np.ones(num_avg), "valid") / num_avg
    return xnew, ynew


def plot_loss_graph(path_to_file: str):
    train_data = []  # iterations, epochs, losses, lm_losses, mc_losses
    test_data = []  # iteration, epoch, mc_correct, num_tests, lm_correct, lm_tested
    with open(path_to_file, "r") as f:
        for line in f:
            split_line = line.strip().split(",")
            if split_line[0] == "train":
                train_data.append(list(map(float, split_line[1:])))
            elif split_line[0] == "test":
                test_data.append(list(map(int, split_line[1:])))

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    mc_accuracy = test_data[:, 2] / test_data[:, 3]
    lm_accuracy = test_data[:, 4] / test_data[:, 5]

    fig, ax = plt.subplots(2, 2)

    ax[0][0].scatter(train_data[:, 0], train_data[:, 2], marker=".", color="blue", alpha=0.01)
    ax[0][0].plot(*moving_avg(train_data[:, 0], train_data[:, 2], 1000), color="black")
    ax[0][0].set_title("Reconstruction loss")
    ax[0][0].set_yscale("log")
    ax[0][1].scatter(train_data[:, 0], train_data[:, 3], marker=".", color="red", alpha=0.01)
    ax[0][1].plot(*moving_avg(train_data[:, 0], train_data[:, 3], 1000), color="black")
    ax[0][1].set_title("Sequence generation loss")
    ax[0][1].set_yscale("log")
    ax[1][0].scatter(train_data[:, 0], train_data[:, 4], marker=".", color="green", alpha=0.01)
    ax[1][0].plot(*moving_avg(train_data[:, 0], train_data[:, 4], 1000), color="black")
    ax[1][0].set_title("Sequence prediction loss")
    ax[1][0].set_yscale("log")
    ax[1][1].scatter(test_data[:, 0], mc_accuracy, color="green")
    ax[1][1].scatter(test_data[:, 0], lm_accuracy, color="red")
    ax[1][1].set_ylim(0, 1)

    plt.show()


if __name__ == "__main__":
    plot_loss_graph("log/log-21-01-01-07-44-37.txt")
