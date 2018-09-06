#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def plot_1():

    epochs = [1, 3, 5, 7, 9, 11, 13, 15]
    mean = [1265.84, 3668.84, 3961.06, 4525.06,
            4634.40, 4794.22, 4680.76, 4780.86]
    std = [590.00, 1184.59, 1288.10, 77.69, 111.45, 126.88, 78.30, 95.73]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, mean)
    plt.errorbar(epochs, mean, yerr=std, fmt='o')
    plt.suptitle('Behavorial Cloning: Epochs vs. Reward', fontsize=15)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean Reward')
    plt.xlim([0, 16])
    plt.legend(["Mean", "Std"])
    plt.savefig("2-2.eps", dpi=300)


def plot_2():

    epochs = [1, 3, 5, 7, 9, 11, 13, 15]
    mean = [-7.70, -5.32, -5.10, -4.64, -5.00, -4.43, -3.71, -3.33]
    std = [2.97, 1.74, 1.52, 1.62, 2.33, 1.89, 1.49, 1.64]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, mean, label="Dagger Mean")
    plt.errorbar(epochs, mean, yerr=std, fmt='o', label="Dagger Std")
    plt.suptitle('DAgger Iterations vs. Rewards', fontsize=15)

    plt.xlabel('DAgger Iteration')
    plt.ylabel('Mean Reward')
    plt.xlim([0, 16])
    plt.ylim([-14, -2])

    plt.axhline(y=-3.78, color='orange', label='Expert')
    plt.axhline(y=-7.11, color='r', label='Behaviorial Cloning')
    plt.legend()
    plt.savefig("3-2.eps", dpi=300)


if __name__ == '__main__':
    plot_1()
    plot_2()
