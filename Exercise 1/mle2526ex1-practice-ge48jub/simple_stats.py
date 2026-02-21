###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2023                                     #
#                                                                         #
#                     Exercise 1 - Template                               #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def readfile(filename, delimiter):
    df = pd.read_csv(filename, delimiter=delimiter, header=None)
    return df.values


def compute_mean(x):
    E = None  # E stands for the expected value
    E = np.sum(x) / x.shape[0]
    return E


def compute_variance(x):
    sigma2 = None  
    mean_x = compute_mean(x)
    mean_x_squared = compute_mean(x**2)
    sigma2 = mean_x_squared - mean_x**2
    return sigma2


def compute_covariance(data):
    N, D = data.shape
    cov = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            xi = data[:, i]
            xj = data[:, j]
            cov[i, j] = np.mean(xi * xj) - np.mean(xi) * np.mean(xj)
    return cov


def main():
     data = readfile("dice.txt",",")
        print(f"Data shape: {data.shape}")

    return 0


if __name__ == "__main__":
    main()
